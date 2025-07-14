from src.sample_attack import SampleAttack
from src.adv_model import AdvModel
from src.activ_extractor import ActivationExtractor, ActivationLoss
from src.eval.evaluator import Evaluator
from src.config import GenConfig
from src.univ_attack import UnivAttack

from typing import Any
import torch
from typing import Iterable, Callable
from functools import partial


# NOTE: currently loss is over all target tokens and not a single token per sample
def cosine_similarity_loss(
    univ_activ: torch.Tensor,
    sample_activ: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        univ_activ (torch.Tensor): Universal activations of shape (batch_size, seq_length, hidden_dim).
        sample_activ (torch.Tensor): Sample activations of shape (batch_size, seq_length, hidden_dim).
        target_mask (torch.Tensor): Mask indicating which tokens are targets of shape (batch_size, seq_length).
    """
    cos_sim = torch.cosine_similarity(univ_activ, sample_activ, dim=-1)  # (batch_size, seq_length)
    loss_matrix = (1 - cos_sim) * target_mask.bool()  # (batch_size, seq_length)
    loss = torch.mean(loss_matrix.sum(dim=-1) / target_mask.sum(dim=-1))
    return loss


# TODO: add scheduling to the inner-attack i.e accept a lambda function that takes the current epoch and
# returns an instance of an inner attack.
# TODO: implmenet discretization
class IML(UnivAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        internal_attack: SampleAttack,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        activ_extractor: ActivationExtractor,
        evaluators: list[Evaluator],
        eval_freq: int | float = 1,
        mixed_precision: bool = True,
        gen_config: GenConfig | None = None,
        skip_already_fooled: bool = False,
        skip_failed_attacks: bool = True,
        dynamic_labels: bool = False,
        log_dir: str | None = None,
    ):
        super().__init__(
            adv_model=adv_model,
            evaluators=evaluators,
            eval_freq=eval_freq,
            mixed_precision=mixed_precision,
            gen_config=gen_config,
            log_dir=log_dir,
        )

        self.internal_attack = internal_attack
        self.activ_extractor = activ_extractor
        self.optimizer = optim_factory([self.univ_embeds])
        self.skip_already_fooled = skip_already_fooled
        self.skip_failed_attacks = skip_failed_attacks
        self.dynamic_labels = dynamic_labels  # TODO: implement dynamic labels

        # TODO: think of a better, less messy way to register hparams
        self.logger.register_hparams({"iml/internal_attack": self.internal_attack.__class__.__name__})
        self.logger.register_hparams({"iml/optimizer": self.optimizer.__class__.__name__})
        self.logger.register_hparams({"iml/skip_already_fooled": self.skip_already_fooled})
        self.logger.register_hparams({"iml/skip_failed_attacks": self.skip_failed_attacks})
        self.logger.register_hparams({"iml/dynamic_labels": self.dynamic_labels})

        self.logger.register_hparams(activ_extractor.get_hparams())
        self.logger.register_hparams({f"internal_attack/{k}": v for k, v in internal_attack.__dict__.items()})
        self.logger.register_hparams({f"optim/name": self.optimizer.__class__.__name__})
        self.logger.register_hparams({f"optim/{k}": v for k, v in self.optimizer.param_groups[0].items()})

    def optim_step(self, data: dict[str, list[Any]], epoch_num: int, batch_num: int) -> float | None:
        # NOTE: IDEA: instead of using a fixed target, generate affirmative responses from the
        # per-sample attacks and use them as targets instead.
        # We should add attack parameter `dynamic_targets` which enables / disables it.
        # CONS:
        # 1. very slow since we need to generate responses, but if we use `skip_failed_attacks=True` then
        # no additional time cost since we generate it anyways.
        # 2. need to be careful with tokenization, tokenize the new generated response as the target
        # PROS:
        # 1. dynamic targets instead of forced ones
        # 2. will allow to support attacks which do not recieve target argument
        # 3. correctness - `skip_failed_attacks` judges the actual generated responses

        # NOTE: IDEA: let the per-sample attacks to also modify the prompt and not only the adversarial tokens.
        # Even more generally - the per-sample attack returns a new adversarial prompt (which may or may not incorporate adv tokens).
        # combined with the previous idea, we then generate an affirmative response to the adver input and use it as the target.
        # CONS:
        # 1. if we allow to modify also the input from the per-sample attack then the affirmative target
        # might not even correspond to the original prompt, therefore its not clear what we're optimizing in that case
        # 2. in that case the returned outputs should be adversarial embeddings and not adversarial input tokens, since SoftPrompt works on the
        # embedding level. Therefore we need to support that.
        # PROS:
        # 1. allows unconstrained use of all per-sample attack altogether:
        #   - for example the per-sample attack doesnt have to use the exact number of adver tokens as the universal attack
        #   - new supported attacks:  direct request attack and also human_jailbreaks, and all attacker-LLM based attacks.

        self.optimizer.zero_grad()

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            # construct conversations
            input_text, target_text = data["prompt"], data["target"]
            conversations = [[{"role": "user", "content": prm}] for prm in input_text]

            # skip already succesfully fooled samples
            if self.skip_already_fooled:
                with torch.inference_mode():
                    self.adv_model.set_embeddings(self.univ_embeds)
                    responses = self.adv_model.chat(conversations, self.gen_config)
                    data["response"] = responses
                    eval_result = self.judge.eval_batch(data)

                    mask_fooled = eval_result >= 1.0
                    if mask_fooled.all():
                        return None

                    conversations = [conv for conv, m in zip(conversations, mask_fooled) if not m]
                    target_text = [tgt for tgt, m in zip(target_text, mask_fooled) if not m]

            # run per-sample attack
            with torch.autocast(device_type=self.device.type, enabled=False):
                init_embeds = self.univ_embeds.expand(len(conversations), -1, -1)
                sample_embed = self.internal_attack.fit(conversations, target_text, init_embeds=init_embeds)
                # TODO: internal attack should return the following:
                # a full conversation (except the targets)
                # if the internal attack is an embedding attack, it should place [adv] tokens in the appropriate
                # places and also return adversarial embeddings for these corresponding places.
                # this way we can add the target to the returned adversarial input and tokenize everything properly.

                # TODO: i think this design removes the need of accepting inputs_embeds parameter in all AdvModel methods
                # i.e we can return the old method design probably.

            # skip failed per-sample attacks
            if self.skip_failed_attacks:
                with torch.inference_mode():
                    self.adv_model.set_embeddings(sample_embed)
                    responses = self.adv_model.chat(conversations, self.gen_config)
                    data["response"] = responses
                    eval_result = self.judge.eval_batch(data)

                    mask_succ = eval_result >= 1.0
                    if not mask_succ.any():
                        return None

                    sample_embed = sample_embed[mask_succ]
                    conversations = [conv for conv, m in zip(conversations, mask_succ) if m]
                    target_text = [tgt for tgt, m in zip(target_text, mask_succ) if m]

            # tokenize remaining coversations
            token_dict = self.adv_model.tokenize(conversations, target_text)

            with self.activ_extractor.capture():
                # compute per-sample activations
                with torch.inference_mode():
                    self.adv_model.set_embeddings(sample_embed)
                    self.adv_model.forward(
                        token_dict["input_ids"], token_dict["attention_mask"], token_dict["adv_mask"]
                    )
                    sample_activs = self.activ_extractor.get_activations()

                # compute universal activations
                self.adv_model.set_embeddings(self.univ_embeds)
                self.adv_model.forward(token_dict["input_ids"], token_dict["attention_mask"], token_dict["adv_mask"])
                univ_activs = self.activ_extractor.get_activations()

            # align target mask and activations
            target_mask = token_dict["target_mask"][:, 1:]
            sample_activs = {k: v[:, :-1] for k, v in sample_activs.items()}
            univ_activs = {k: v[:, :-1] for k, v in univ_activs.items()}

            # compute loss
            criterion = ActivationLoss(loss_fn=partial(cosine_similarity_loss, target_mask=target_mask))
            loss = criterion.forward(univ_activs, sample_activs)

        # grad step
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return loss.item()
