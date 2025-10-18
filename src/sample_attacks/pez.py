from src.adv_model import AdvModel
from src.sample_attacks.soft_prompt import SoftPrompt
from src.sample_attacks.sample_attack import SampleOutput
from src.discretize import Discretize
from src.initialize import Initializer
from src.aliases import Conv

import inspect
from typing import Callable, Iterable
from tqdm.auto import tqdm
import torch
import copy
from torch import nn
from transformers.cache_utils import DynamicCache


def default_optimizer(params: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=0.001, weight_decay=0.0)


class SoftProject(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        discretize_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        """
        A module that projects soft embeddings to the nearest hard embeddings
        in the forward pass and passes the gradient through in the backward pass.

        Args:
            embedding (nn.Embedding): The embedding layer to use for projection.
            discretize_func (Callable): A function that takes in soft embeddings and the embedding
                weight matrix and returns the nearest neighbor ids.
        """

        super().__init__()
        self.embedding = embedding
        self.discretize_func = discretize_func

        class _STE(torch.autograd.Function):
            @staticmethod
            @torch.no_grad()
            def forward(ctx, soft_embeds: torch.Tensor):
                ids = self.discretize(soft_embeds)
                return self.embedding(ids)

            @staticmethod
            def backward(ctx, *grad_outputs: torch.Tensor):
                return grad_outputs

        self._STE = _STE

    def get_hparams(self) -> dict:
        return {
            "num_embeddings": self.embedding.num_embeddings,
            "discretize_func": inspect.getsource(self.discretize_func),
        }

    @torch.no_grad()
    def discretize(self, soft_embeds: torch.Tensor) -> torch.Tensor:
        """
        Return the discrete token ids corresponding to the nearest hard embeddings.

        Args:
            soft_embeds (torch.Tensor): Soft embeddings of shape [b, n, d]

        Returns:
            ids (torch.Tensor): Nearest neighbor of each soft embedding in the vocabulary, shape [b, n]
        """
        return self.discretize_func(soft_embeds, self.embedding.weight)

    def forward(self, soft_embeds: torch.Tensor) -> torch.Tensor:
        """
        Project the soft embeddings to the nearest hard embeddings in the forward pass
        and pass the gradient through in the backward pass.
        """
        return self._STE.apply(soft_embeds)  # type: ignore


class PEZ(SoftPrompt):
    """
    PEZ attack implementation matching the one from Harm-Bench
    (see https://github.com/centerforaisafety/HarmBench/blob/main/baselines/pez/pez.py).
    """

    def __init__(
        self,
        adv_model: AdvModel,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer] = default_optimizer,
        steps: int = 100,
        early_stopping: bool = False,
        mixed_precision: bool = False,
        return_embeds: bool = False,  # TODO: compare both True and False results; change to True after testing
        verbose: bool = True,
    ):
        super().__init__(
            adv_model=adv_model,
            optim_factory=optim_factory,
            steps=steps,
            early_stopping=early_stopping,
            mixed_precision=mixed_precision,
            verbose=verbose,
        )

        self.return_embeds = return_embeds
        self.soft_project = SoftProject(self.adv_model.orig_embedder, Discretize.cosine_similarity)

    def get_hparams(self) -> dict:
        hparams = super().get_hparams()
        hparams.update(
            {
                "return_embeds": self.return_embeds,
                "soft_project": self.soft_project.get_hparams(),
            }
        )
        return hparams

    def _initialize_embeddings(
        self,
        num_inputs: int,
        init_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if init_embeds is None:
            init_embeds = Initializer.from_random_ids(
                self.adv_model,
                allow_nonascii=True,
                allow_special=True,
                batch_size=num_inputs,
            )

        init_embeds = init_embeds.clone().detach()
        init_embeds.requires_grad_(True)
        return init_embeds

    def fit(
        self,
        conversations: list[Conv],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> SampleOutput:
        conversations = copy.deepcopy(conversations)
        
        # HB impl. adds space before adversarial tokens
        for conv in conversations:
            conv[-1]["content"] = conv[-1]["content"] + " "

        # initialize optimized embeddings
        adv_embeds = self._initialize_embeddings(
            num_inputs=len(conversations),
            init_embeds=init_embeds,
        )

        # create optimizer and scaler
        scaler = torch.GradScaler(enabled=self.mixed_precision)
        optim = self.optim_factory([adv_embeds])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.steps)

        # tokenize
        conversations = self.adv_model.inject_tokens(conversations, add_spaces=False, adv_suffix=True)
        encodings = self.adv_model.tokenize(conversations, target_texts)

        # compute kv-cache
        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            encodings = self._compute_cache(encodings)
            kv_cache: DynamicCache = encodings.kv_cache
            cache_length = kv_cache.get_seq_length()

        # early stopping state
        finished = torch.zeros(len(conversations), dtype=torch.bool, device=self.device)
        optim_embeds = adv_embeds

        with tqdm(range(self.steps), disable=not self.verbose, leave=False, desc="Attack") as pbar:
            for step in pbar:
                optim.zero_grad()

                # NOTE: need to crop kv-cache since forward modifies it in-place
                step_encodings = encodings.copy()
                step_encodings["kv_cache"] = kv_cache.crop(cache_length)

                # select only unfinished samples if early stopping is enabled
                if self.early_stopping:
                    step_encodings = self._masked_select(step_encodings, ~finished)
                    optim_embeds = adv_embeds[~finished]

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    # forward pass
                    optim_embeds = self.soft_project.forward(optim_embeds)

                    adv_content = self.adv_model.forward(
                        input_ids=step_encodings.input_ids,
                        attention_mask=step_encodings.attention_mask,
                        adv_mask=step_encodings.adv_mask,
                        past_key_values=step_encodings.kv_cache,
                        adv_embeds=optim_embeds,
                    )

                    # align predicted logits and target_ids
                    logits: torch.Tensor = adv_content.logits[:, :-1]  # remove new token
                    target_ids = step_encodings.input_ids[:, 1:]  # remove BOS token
                    target_mask = step_encodings.target_mask[:, 1:]  # remove BOS token

                    # update early stopping based on predictions
                    if self.early_stopping:
                        finished_status = self._check_early_stopping(logits, target_ids, target_mask)
                        finished[~finished] = finished_status
                        if finished.all():
                            # break early
                            pbar.n = pbar.total
                            pbar.close()
                            break
                        
                        # logits = logits[~finished_status]
                        # target_ids = target_ids[~finished_status]
                        # target_mask = target_mask[~finished_status]

                    loss = self.criterion(
                        logits=logits,
                        target_ids=target_ids,
                        target_mask=target_mask,
                    )

                # backward pass and optimization step
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                sched.step()

                pbar.set_postfix({"loss": loss.item()})

        with torch.no_grad():
            # replace adv token placeholders with discrete tokens
            adv_ids = self.soft_project.discretize(adv_embeds)

            if self.return_embeds:
                # return convs with adv-token placeholders and the discrete embeddings
                discrete_embeds = self.adv_model.embed(adv_ids)
                return SampleOutput(conversations, discrete_embeds.detach())

        # replace adv token placeholders with discrete tokens
        conversations = self.adv_model.repl_tokens(conversations, repl_ids=adv_ids.tolist())
        return SampleOutput(conversations)
