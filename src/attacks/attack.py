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
