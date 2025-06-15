from src.adver_model import AdverModel
import torch
from abc import ABC, abstractmethod
import copy


class Attack(ABC):
    def __init__(
        self,
        adv_model: AdverModel,
        verbose: bool = True,
    ):
        self.adv_model = adv_model
        self.verbose = verbose

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
        return self.adv_model.adv_embedder.embed_dim

    @property
    def embed_dtype(self) -> torch.dtype:
        """
        Returns the embedding dtype of the model.
        """
        return self.adv_model.adv_embedder.embed_dtype

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

    @torch.no_grad()
    def compute_cache(self, token_dict: dict[str, torch.Tensor]) -> dict:
        """
        Compute the kv-cache for the constant part of the input.

        Args:
            token_dict (dict[str, torch.Tensor]): Dictionary containing input tokens and masks,
                as returned from `self.adv_model.tokenize()`.

        Returns:
            dict: Dictionary containing the kv-cache for the constant part of the input,
                as well as the remaining input tokens and masks. The remaining input tokens
                and masks are only the variable (non-cached) part of the input.
        """

        kv_idx = token_dict["const_idx"].min().item()

        kv_result = self.adv_model.forward(
            token_dict["input_ids"][:, :kv_idx],
            token_dict["attention_mask"][:, :kv_idx],
            use_cache=True,
        )

        kv_cache = copy.deepcopy(kv_result.past_key_values)

        new_token_dict = {
            "input_ids": token_dict["input_ids"][:, kv_idx:],
            "attention_mask": token_dict["attention_mask"],  # we need the full attention mask
            "adv_mask": token_dict["adv_mask"][:, kv_idx:],
            "const_idx": token_dict["const_idx"],
            "kv_cache": kv_cache,
        }

        if "target_mask" in token_dict:
            new_token_dict["target_mask"] = token_dict["target_mask"][:, kv_idx:]

        return new_token_dict

    @abstractmethod
    def fit(
        self,
        conversations: list[list[dict[str, str]]],
        target_texts: list[str],
        embeds_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fit the attack model to the input and target texts.

        Args:
            conversations (list[list[dict[str, str]]]): List of conversations, where each conversation is a list of messages.
                Each message is a dictionary with keys "role" and "content".
            target_texts (list[str]): List of target texts.
            embeds_init (torch.Tensor | None): Initial adversarial embedding.

        Returns:
            torch.Tensor: Adversarial embedding of shape (batch_size, num_tokens, embedding_dim).
        """
        raise NotImplementedError("Subclasses must implement this method.")
