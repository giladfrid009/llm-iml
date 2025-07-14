from src.sample_attacks.harm_bench.baseline import SequentialHarmBenchAttack
from src.sample_attacks.harm_bench.utils import get_template
from src.adv_model import AdvModel

import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
import numpy as np
from copy import deepcopy
from accelerate.utils import find_executable_batch_size


# ============================== UTILS FOR UAT ============================== #
def hotflip_attack(
    averaged_grad: torch.Tensor,
    embedding_matrix: torch.Tensor,
    trigger_token_ids,
    increase_loss: bool = False,
    num_candidates: int = 1,
) -> np.ndarray | int:
    """
    The "Hotflip" attack described in Equation (2) of the paper. This code is heavily inspired by
    the nice code of Paul Michel here https://github.com/pmichel31415/translate/blob/paul/
    pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py

    This function takes in the model's average_grad over a batch of examples, the model's
    token embedding matrix, and the current trigger token IDs. It returns the top token
    candidates for each position.

    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).
    """
    averaged_grad = averaged_grad
    embedding_matrix = embedding_matrix
    trigger_token_embeds = (
        torch.nn.functional.embedding(torch.LongTensor(trigger_token_ids).cuda(), embedding_matrix)
        .detach()
        .unsqueeze(0)
    )  # TODO: WHY IS IT UNUSED?
    averaged_grad = averaged_grad.unsqueeze(0)

    # einsum does not support half precision, so we rewrite as bmm.
    gradient_dot_embedding_matrix = torch.bmm(averaged_grad, torch.transpose(embedding_matrix, 0, 1).unsqueeze(0))

    if not increase_loss:
        gradient_dot_embedding_matrix *= -1  # lower versus increase the class probability.
    if num_candidates > 1:  # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()


# ============================== GCG CLASS DEFINITION ============================== #
class UAT(SequentialHarmBenchAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        num_steps=50,
        adv_string_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        use_prefix_cache=True,
        num_candidates=3,
        verbose: bool = True,
    ):
        """
        :param adv_model:
        :param num_steps: the number of optimization steps to use
        :param adv_string_init: the initial adversarial string to start with
        :param use_prefix_cache: whether to use KV cache for static prefix tokens (WARNING: Not all models support this)
        :param num_candidates: the number of candidates to consider at each step
        """
        super().__init__(adv_model, verbose)
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.use_prefix_cache = use_prefix_cache
        self.num_candidates = num_candidates

        template = get_template(self.tokenizer)
        self.template = template
        self.before_tc, self.after_tc = template.split("{instruction}")

    def extract_grad_hook(self, module, grad_in, grad_out):
        self.extracted_grads.append(grad_out[0])

    def get_loss(
        self,
        search_batch_size: int,
        input_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        target_ids: torch.Tensor,
        no_grad: bool = True,
    ) -> torch.Tensor:
        loss_list = []
        for i in range(0, input_embeds.shape[0], search_batch_size):
            input_embeds_batch = input_embeds[i : i + search_batch_size]
            current_batch_size = input_embeds_batch.shape[0]

            # forward pass
            if self.use_prefix_cache:
                # Expand prefix cache to match batch size
                prefix_cache_batch = []
                for i in range(len(self.prefix_cache)):
                    prefix_cache_batch.append([])
                    for j in range(len(self.prefix_cache[i])):
                        prefix_cache_batch[i].append(self.prefix_cache[i][j].expand(current_batch_size, -1, -1, -1))
                if no_grad:
                    with torch.no_grad():
                        outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch)
            else:
                if no_grad:
                    with torch.no_grad():
                        outputs = self.model(inputs_embeds=input_embeds_batch)
                else:
                    outputs = self.model(inputs_embeds=input_embeds)

            logits: torch.Tensor = outputs.logits
            # compute loss
            # Shift so that tokens < n predict n
            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
            shift_labels = target_ids.repeat(current_batch_size, 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct.forward(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(current_batch_size, -1).mean(dim=1)
            loss_list.append(loss)
        loss = torch.cat(loss_list, dim=0)
        return loss

    def generate_test_cases_single_behavior(self, behavior: str, target: str) -> str:
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: a test case and logs
        """
        # starting search_batch_size, will auto reduce batch_size later if go OOM (resets for every new behavior)
        self.search_batch_size = self.num_candidates

        ### Targeted Model and Tokenier ###
        model = self.model
        device = model.device
        tokenizer = self.tokenizer

        ### vars ###
        adv_string_init = self.adv_string_init
        num_steps = self.num_steps
        before_tc = self.before_tc
        after_tc = self.after_tc

        # ========= add gradient hooks to embeddings =========
        embed_layer: torch.nn.Embedding = model.get_input_embeddings()
        embed_layer.weight.requires_grad = True
        embed_layer.register_backward_hook(self.extract_grad_hook)
        embedding_weight = embed_layer.weight.detach()

        # ========== setup trigger_tokens and input_embeds components ========== #
        trigger_tokens = tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)[
            "input_ids"
        ][0]
        trigger_token_length = len(trigger_tokens)

        behavior_ids = tokenizer(behavior, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]
        behavior_embeds = embed_layer(behavior_ids)
        target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]
        target_embeds = embed_layer(target_ids)

        before_ids = tokenizer(before_tc, return_tensors="pt").to(device)["input_ids"]
        before_embeds = embed_layer(before_ids)
        after_ids = tokenizer(after_tc, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]
        after_embeds = embed_layer(after_ids)

        if self.use_prefix_cache:
            # precompute and cache states for system prompt and user prompt
            init_input_embeds = torch.cat([before_embeds, behavior_embeds], dim=1)
            with torch.no_grad():
                outputs = model(inputs_embeds=init_input_embeds, use_cache=True)
                self.prefix_cache = outputs.past_key_values

        # ========== get initial loss for the trigger ==========
        model.zero_grad()
        if self.use_prefix_cache:
            input_embeds = torch.cat([embed_layer(trigger_tokens.unsqueeze(0)), after_embeds, target_embeds], dim=1)
        else:
            input_embeds = torch.cat(
                [before_embeds, behavior_embeds, embed_layer(trigger_tokens.unsqueeze(0)), after_embeds, target_embeds],
                dim=1,
            )
        loss = self.get_loss(1, input_embeds, target_embeds, target_ids, no_grad=False)
        best_loss = loss
        counter = 0
        end_iter = False

        # ========== run optimization ========== #
        for i in tqdm(range(num_steps), desc=f"UAT"):
            if end_iter:
                break

            for token_to_flip in range(0, trigger_token_length):
                if end_iter:
                    break
                # Get average gradient w.r.t. the triggers
                extracted_grads = []  # clear the gradient from past iterations # TODO: WHY IS IT UNUSED?
                self.extracted_grads = []
                loss.backward(retain_graph=True)

                averaged_grad = torch.sum(self.extracted_grads[0], dim=0)
                averaged_grad = averaged_grad[token_to_flip].unsqueeze(0)

                # Use hotflip (linear approximation) attack to get the top num_candidates
                candidates = hotflip_attack(
                    averaged_grad,
                    embedding_weight,
                    [trigger_tokens[token_to_flip]],
                    increase_loss=False,
                    num_candidates=self.num_candidates,
                )[0]

                # Prepare a batch of candidate embeddings
                candidates_tokens = []
                for cand in candidates:
                    candidate_trigger_tokens = deepcopy(trigger_tokens)
                    candidate_trigger_tokens[token_to_flip] = cand
                    candidates_tokens.append(candidate_trigger_tokens)
                candidates_tokens = torch.stack(candidates_tokens, dim=0)

                candidates_embeds = embed_layer(candidates_tokens)
                if self.use_prefix_cache:
                    input_embeds_candidates = torch.cat(
                        [
                            candidates_embeds,
                            after_embeds.repeat(len(candidates), 1, 1),
                            target_embeds.repeat(len(candidates), 1, 1),
                        ],
                        dim=1,
                    )
                else:
                    input_embeds_candidates = torch.cat(
                        [
                            before_embeds.repeat(len(candidates), 1, 1),
                            behavior_embeds.repeat(len(candidates), 1, 1),
                            candidates_embeds,
                            after_embeds.repeat(len(candidates), 1, 1),
                            target_embeds.repeat(len(candidates), 1, 1),
                        ],
                        dim=1,
                    )

                # Auto Find Batch Size for foward candidates (each time go OOM will decay search_batch_size // 2)
                candidate_losses = find_executable_batch_size(self.get_loss, self.search_batch_size)(
                    input_embeds_candidates, target_embeds, target_ids
                )

                curr_best_loss = candidate_losses[candidate_losses.argmin()]
                curr_best_trigger_tokens = candidates_tokens[candidate_losses.argmin()]

                # Update overall best if the best current candidate is better
                if curr_best_loss < best_loss:
                    counter = 0  # used to exit early if no improvements in the trigger
                    best_loss = curr_best_loss
                    trigger_tokens = curr_best_trigger_tokens
                # if you have gone through all trigger_tokens without improvement, end iteration
                elif counter == len(trigger_tokens):
                    end_iter = True
                # If the loss didn't get better, just move to the next word.
                else:
                    counter = counter + 1
                # reevaluate the best candidate so you can backprop into it at next iteration
                model.zero_grad()

                if self.use_prefix_cache:
                    input_embeds = torch.cat(
                        [embed_layer(trigger_tokens.unsqueeze(0)), after_embeds, target_embeds], dim=1
                    )
                else:
                    input_embeds = torch.cat(
                        [
                            before_embeds,
                            behavior_embeds,
                            embed_layer(trigger_tokens.unsqueeze(0)),
                            after_embeds,
                            target_embeds,
                        ],
                        dim=1,
                    )
                loss = self.get_loss(1, input_embeds, target_embeds, target_ids, no_grad=False)

            test_case_ids = torch.cat([behavior_ids, trigger_tokens.unsqueeze(0)], dim=1)
            test_case = tokenizer.decode(test_case_ids[0])

        return test_case
