from src.adv_model import AdvModel
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SampleOutput:
    conversations: list[list[dict[str, str]]]
    adv_embeds: torch.Tensor | None = None


class SampleAttack(ABC):
    def __init__(
        self,
        adv_model: AdvModel,
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

    @abstractmethod
    def fit(
        self,
        conversations: list[list[dict[str, str]]],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> SampleOutput:
        """
        Fit the attack model to the input and target texts.

        Args:
            conversations (list[list[dict[str, str]]]): List of conversations, where each conversation is a list of messages.
                Each message is a dictionary with keys "role" and "content".
            target_texts (list[str]): List of target texts.
            init_embeds (torch.Tensor | None): Initial embeddings to use for the attack.

        Returns:
            SampleOutput: An object containing adversarial conversations, and optionally adversarial embeddings.
        """
        raise NotImplementedError("Subclasses must implement this method.")
