from src.adver_model import AdverModel
import torch
from abc import ABC, abstractmethod


class Attack(ABC):
    def __init__(
        self,
        adv_model: AdverModel,
        verbose: bool = True,
    ):
        self.adv_model = adv_model
        self.verbose = verbose

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model is located.
        """
        return self.adv_model.device

    # TODO: should probably modify the signature if this method
    # not all attacks need or use target_texts
    # the signature should support all LLM attack types.
    
    # i think its fine to recieve a dictionary of conversations as all attack methods attack
    # chat models 
    # its fine to assume the conversations are already well formatted and the user prompt is the full prompt
    @abstractmethod
    def fit(
        self,
        conversations: list[list[dict[str, str]]],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fit the attack model to the input and target texts.

        Args:
            conversations (list[list[dict[str, str]]]): List of conversations, where each conversation is a list of messages.
                Each message is a dictionary with keys "role" and "content".
            target_texts (list[str]): List of target texts.
            init_embeds (torch.Tensor | None): Initial adversarial embedding.

        Returns:
            torch.Tensor: Adversarial embedding of shape (batch_size, num_tokens, embedding_dim).
        """
        raise NotImplementedError("Subclasses must implement this method.")
