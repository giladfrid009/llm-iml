from src.adver_model import AdverModel
import torch
from abc import ABC, abstractmethod


class Attack(ABC):
    def __init__(
        self,
        adv_model: AdverModel,
        silent: bool = False,
    ):
        self.adv_model = adv_model
        self.silent = silent

    @property
    def num_tokens(self) -> int:
        """
        Returns the number of adversarial tokens to be injected.
        """
        return self.adv_model.num_tokens

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model is located.
        """
        return self.adv_model.device

    @property
    def embed_dim(self) -> int:
        """
        Returns the embedding dimension of the model.
        """
        return self.adv_model.embed_dim

    @property
    def embed_dtype(self) -> torch.dtype:
        """
        Returns the embedding dtype of the model.
        """
        return self.adv_model.embed_dtype

    def align_preds(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aligns the predictions with the target tokens for loss calculation.

        Args:
            logits (torch.Tensor): Model output logits.
            input_ids (torch.Tensor): Input token IDs.
            target_mask (torch.Tensor): Mask indicating target tokens.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing aligned logits and target IDs.
        """
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        target_mask = target_mask[:, 1:]

        target_ids = input_ids[target_mask].view(-1)
        pred_logits = logits[target_mask].view(-1, logits.size(-1))
        return pred_logits, target_ids

    def init_embedding(self, num_inputs: int, **kwargs) -> torch.Tensor:
        return torch.randn(
            size=(num_inputs, self.num_tokens, self.embed_dim),
            device=self.device,
            dtype=self.embed_dtype,
            requires_grad=True,
        )

    @abstractmethod
    def fit(
        self,
        input_texts: list[str],
        target_texts: list[str],
        adv_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fit the attack model to the input and target texts.

        Args:
            input_texts (list[str]): List of input texts.
            target_texts (list[str]): List of target texts.
            adv_embeds (torch.Tensor | None): Initial adversarial embedding.

        Returns:
            torch.Tensor: Adversarial embedding of shape (batch_size, num_tokens, embedding_dim).
        """
        raise NotImplementedError("Subclasses must implement this method.")
