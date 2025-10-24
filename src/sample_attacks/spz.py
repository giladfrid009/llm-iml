from src.adv_model import AdvModel
from src.sample_attacks.sp import SP
from src.sample_attacks.base import SampleOutput
from src.initialize import Initializer
from src.aliases import Conv

import inspect
from typing import Callable, Iterable
import torch


def default_injector(adv_model: AdvModel, conversations: list[Conv]) -> list[Conv]:
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
    return Initializer.random_normal(adv_model, std=0.1, batch_size=num_inputs)


class SPZ(SP):
    """
    Soft Prompt-Zero attack with custom adversarial token injection and initialization functions.
    This variant always initializes new adversarial embeddings from scratch,
    ignoring any provided initial embeddings.
    """

    def __init__(
        self,
        adv_model: AdvModel,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        inject_func: Callable[[AdvModel, list[Conv]], list[Conv]] = default_injector,
        init_func: Callable[[AdvModel, int], torch.Tensor] = default_initializer,
        **kwargs,
    ):
        """
        Args:
            adv_model (AdvModel): The adversarial model to attack.
            optim_factory (Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]): Returns an optimizer given the parameters to optimize.
            inject_func (Callable[[AdvModel, list[Conv]], list[Conv]]): Performs adversarial token injection to the conversations.
            init_func (Callable[[AdvModel, int], torch.Tensor]): Initializes new adversarial embeddings from scratch.
            **kwargs: Additional arguments passed to the `sample_attacks.SP` constructor.
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
        conversations: list[Conv],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> SampleOutput:
        # inject adversarial tokens according to custom function
        inj_conversations = self.inject_func(self.adv_model, conversations)

        # initialize new embeddings
        init_embeds = self.init_func(self.adv_model, len(conversations))

        return super().fit(
            inj_conversations,
            target_texts,
            init_embeds=init_embeds,
        )
