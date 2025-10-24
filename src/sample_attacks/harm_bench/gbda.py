from src.sample_attacks.harm_bench.base import SequentialHarmBenchAttack
from src.sample_attacks.harm_bench.utils import get_template
from src.adv_model import AdvModel


import torch
from torch.nn import CrossEntropyLoss
import numpy as np


# ============================== GBDA CLASS DEFINITION ============================== #
class GBDA(SequentialHarmBenchAttack):
    """
    Gradient-based Distributional Attack
    [https://arxiv.org/pdf/2104.13733]
    """

    def __init__(
        self,
        adv_model: AdvModel,
        num_optim_tokens=20,
        num_steps=50,
        lr=0.2,
        noise_scale=0.2,
        verbose: bool = True,
    ):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param num_optim_tokens: the number of tokens in each test case
        :param num_steps: the number of optimization steps to use
        :param lr: the learning rate to use
        :param noise_scale: amount of noise to use for random initialization
        """
        super().__init__(adv_model, verbose)
        self.num_optim_tokens = num_optim_tokens
        self.num_steps = num_steps
        self.lr = lr
        self.noise_scale = noise_scale

        template = get_template(self.tokenizer)
        self.template = template
        self.before_tc, self.after_tc = template.split("{instruction}")

    def get_hparams(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "num_optim_tokens": self.num_optim_tokens,
            "num_steps": self.num_steps,
            "lr": self.lr,
            "noise_scale": self.noise_scale,
        }

    def generate_test_cases_single_behavior(self, behavior: str, target: str) -> str:
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param num_generate: the number of test cases to generate in parallel
        :param verbose: whether to print progress
        :return: a list of test case and a list of logs
        """
        # ========== Behavior and Target str ==========
        behavior += " "

        # GBDA hyperparams
        num_generate = 1
        num_optim_tokens = self.num_optim_tokens
        num_steps = self.num_steps
        lr = self.lr
        noise_scale = self.noise_scale

        # Vars
        model = self.model
        tokenizer = self.tokenizer
        device = model.device
        before_tc = self.before_tc
        after_tc = self.after_tc
        embed_layer: torch.nn.Embedding = self.model.get_input_embeddings()

        # ========== Init Cache Embeds ========
        cache_input_ids = tokenizer([before_tc], padding=False)["input_ids"]  # some tokenizer have <s> for before_tc
        cache_input_ids += tokenizer([behavior, after_tc, target], padding=False, add_special_tokens=False)["input_ids"]
        cache_input_ids = [
            torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids
        ]  # make tensor separately because can't return_tensors='pt' in tokenizer
        before_ids, behavior_ids, after_ids, target_ids = cache_input_ids
        before_embeds, behavior_embeds, after_embeds, target_embeds = [
            embed_layer(input_ids) for input_ids in cache_input_ids
        ]

        # ========== setup log_coeffs (the optimizable variables) ========== #
        with torch.no_grad():
            embeddings = embed_layer(torch.arange(0, self.tokenizer.vocab_size, device=device).long())

        # ========== setup log_coeffs (the optimizable variables) ========== #
        log_coeffs = torch.zeros(num_generate, num_optim_tokens, embeddings.size(0))
        log_coeffs += torch.randn_like(log_coeffs) * noise_scale  # add noise to initialization
        log_coeffs = log_coeffs.cuda()
        log_coeffs.requires_grad_()

        # ========== setup optimizer and scheduler ========== #
        optimizer = torch.optim.Adam([log_coeffs], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)
        taus = np.linspace(1, 0.1, num_steps)

        # ========== run optimization ========== #
        for i in range(num_steps):
            coeffs = torch.nn.functional.gumbel_softmax(log_coeffs, hard=False, tau=taus[i]).to(
                embeddings.dtype
            )  # B x T x V
            optim_embeds = coeffs @ embeddings[None, :, :]  # B x T x D

            input_embeds = torch.cat(
                [
                    before_embeds.repeat(num_generate, 1, 1),
                    behavior_embeds.repeat(num_generate, 1, 1),
                    optim_embeds,
                    after_embeds.repeat(num_generate, 1, 1),
                    target_embeds.repeat(num_generate, 1, 1),
                ],
                dim=1,
            )

            outputs = model(inputs_embeds=input_embeds)
            logits: torch.Tensor = outputs.logits

            # ========== compute loss ========== #
            # Shift so that tokens < n predict n
            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
            shift_labels = target_ids.repeat(num_generate, 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct.forward(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(num_generate, -1).mean(dim=1)

            # ========== update log_coeffs ========== #
            optimizer.zero_grad()
            loss.mean().backward(inputs=[log_coeffs])
            optimizer.step()
            scheduler.step()

            # ========== retrieve optim_tokens and test_cases========== #
            optim_tokens = torch.argmax(log_coeffs, dim=2)
            test_cases_tokens = torch.cat([behavior_ids.repeat(num_generate, 1), optim_tokens], dim=1)
            test_cases = tokenizer.batch_decode(test_cases_tokens, skip_special_tokens=True)

        return test_cases[0]
