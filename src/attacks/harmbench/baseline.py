from tqdm.auto import tqdm

from src.adver_model import AdverModel
from src.attacks.attack import Attack
import torch


class RedTeamingMethod(Attack):
    """
    A template for a red teaming method that generates test cases given a set of behaviors
    """

    def __init__(self, adv_mode: AdverModel, verbose: bool):
        self.adv_model = adv_mode
        self.verbose = verbose

        # model and tokenizer
        self.model = adv_mode.model
        self.tokenizer = adv_mode.tokenizer

    def generate_test_cases(self, behaviors: list[str], targets: list[str], init_embeds: torch.Tensor | None = None) -> list[str]:
        raise NotImplementedError

    def fit(
        self,
        conversations: list[list[dict[str, str]]],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if len(conversations) != len(target_texts):
            raise ValueError("The number of conversations must match the number of target texts.")

        assert all(len(conv) == 1 for conv in conversations)
        assert all(conv[-1]["role"] == "user" for conv in conversations)

        behaviors = [conv[-1]["content"] for conv in conversations]
        test_cases = self.generate_test_cases(behaviors, target_texts, init_embeds)

        # TODO: IMPLEMENT, what is test_cases even?
        raise NotImplementedError("Subclasses must implement this method.")


class SingleBehaviorRedTeamingMethod(RedTeamingMethod):
    """
    A template method that generates test cases for a single behavior and model
    (e.g., used by GCG, PEZ, GBDA, UAT, AutoPrompt)
    """

    def __init__(
        self,
        adv_model: AdverModel,
        verbose: bool,
    ):
        """ """
        super().__init__(adv_model, verbose)

    def generate_test_cases(self, behaviors: list[str], targets: list[str], init_embeds: torch.Tensor | None = None) -> list[str]:

        if len(behaviors) != len(targets):
            raise ValueError("The number of behaviors must match the number of targets.")

        test_cases = []
        for beh, tgt in tqdm(zip(behaviors, targets), disable=not self.verbose):
            test_case = self.generate_test_cases_single_behavior(beh, tgt)
            test_cases.append(test_case)
        return test_cases

    def generate_test_cases_single_behavior(self, behavior: str, target: str) -> str:
        raise NotImplementedError
