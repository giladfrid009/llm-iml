from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.aliases import Conv
from src.adv_model import AdvModel


@dataclass
class SampleOutput:
    """
    Note:
        If `adv_embeds` is not None, the `conversations` are expected to contain adversarial tokens in the appropriate places,
        to be replaced by the adversarial embeddings during model forward.
    """

    conversations: list[Conv]
    adv_embeds: torch.Tensor | None = None

    def masked_select(self, mask: torch.Tensor) -> SampleOutput:
        convs = [conv for conv, m in zip(self.conversations, mask) if m]
        adv_embeds = self.adv_embeds[mask] if self.adv_embeds is not None else None
        return SampleOutput(convs, adv_embeds)


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
        conversations: list[Conv],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> SampleOutput:
        """
        Fit the attack model to the input and target texts.

        Args:
            conversations (list[Conv]): List of conversations, where each conversation is a list of messages.
            target_texts (list[str]): List of target texts.
            init_embeds (torch.Tensor | None): Initial embeddings to use for the attack.

        Returns:
            SampleOutput: An object containing adversarial conversations, and optionally adversarial embeddings.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_hparams(self) -> dict:
        """
        Returns a dictionary of hyperparameters for the attack.
        """
        raise NotImplementedError("Subclasses must implement this method.")
