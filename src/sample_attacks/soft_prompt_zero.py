from src.adv_model import AdvModel
from src.sample_attacks.soft_prompt import SoftPrompt
from src.sample_attacks.sample_attack import SampleOutput

import inspect
from typing import Callable, Iterable
import torch


Convs = list[list[dict[str, str]]]


def default_injector(adv_model: AdvModel, conversations: Convs) -> Convs:
    """
    Injects an adversarial prefix to each conversation in the list,
    with spaces separating the adversarial tokens.
    """
    return adv_model.inject_tokens(
        conversations,
        add_spaces=True,
        adv_suffix=False,
    )


def default_initializer(adv_model: AdvModel, num_inputs: int) -> torch.Tensor:
    """
    Random normal initialization with standard deviation of 0.1.
    """
    embeddings = torch.randn(
        size=(num_inputs, adv_model.num_tokens, adv_model.adv_embedder.embed_dim),
        device=adv_model.device,
        dtype=adv_model.adv_embedder.embed_dtype,
    )

    return embeddings * 0.1


class SoftPromptZero(SoftPrompt):
    """
    SoftPrompt attack with custom adversarial token injection and initialization functions.
    This variant always initializes new adversarial embeddings from scratch,
    ignoring any provided initial embeddings.
    """

    def __init__(
        self,
        adv_model: AdvModel,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        inject_func: Callable[[AdvModel, Convs], Convs] = default_injector,
        init_func: Callable[[AdvModel, int], torch.Tensor] = default_initializer,
        **kwargs,
    ):
        """
        Args:
            adv_model (AdvModel): The adversarial model to attack.
            optim_factory (Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]): Returns an optimizer given the parameters to optimize.
            inject_func (Callable[[AdvModel, Convs], Convs]): Performs adversarial token injection to the conversations.
            init_func (Callable[[AdvModel, int], torch.Tensor]): Initializes new adversarial embeddings from scratch.
            **kwargs: Additional arguments passed to the `SoftPrompt` constructor.
        """
        super().__init__(
            adv_model=adv_model,
            optim_factory=optim_factory,
            **kwargs,
        )

        self.inject_func = inject_func
        self.init_func = init_func

    def get_hparams(self) -> dict:
        hparams = super().get_hparams()
        hparams.update(
            {
                "inject_func": inspect.getsource(self.inject_func),
                "init_func": inspect.getsource(self.init_func),
            }
        )
        return hparams

    def fit(
        self,
        conversations: list[list[dict[str, str]]],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> SampleOutput:
        # inject adversarial tokens according to custom function
        inj_conversations = self.inject_func(self.adv_model, conversations)

        # initialize new embeddings
        init_embeds = self.init_func(self.adv_model, len(conversations))
        init_embeds.requires_grad_(True)

        return super().fit(
            inj_conversations,
            target_texts,
            init_embeds=init_embeds,
        )
