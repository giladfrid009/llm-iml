from tqdm.auto import tqdm

from abc import abstractmethod
from src.adv_model import AdvModel
from src.sample_attack import TextAttack
import torch
import copy


class HarmBenchAttack(TextAttack):
    """
    A template for a red teaming method that generates test cases given a set of behaviors
    """

    def __init__(self, adv_mode: AdvModel, verbose: bool):
        self.adv_model = adv_mode
        self.verbose = verbose

        # model and tokenizer
        self.model = adv_mode.model
        self.tokenizer = adv_mode.tokenizer

    @abstractmethod
    def generate_test_cases(
        self, behaviors: list[str], targets: list[str], init_embeds: torch.Tensor | None = None
    ) -> list[str]:
        raise NotImplementedError

    # TODO: what about init_embeds?
    def fit_text(
        self,
        conversations: list[list[dict[str, str]]],
        target_texts: list[str],
    ) -> list[list[dict[str, str]]]:

        if len(conversations) != len(target_texts):
            raise ValueError("The number of conversations must match the number of target texts.")

        assert all(len(conv) == 1 for conv in conversations)
        assert all(conv[-1]["role"] == "user" for conv in conversations)

        behaviors = [conv[-1]["content"] for conv in conversations]

        # TODO: what about init_embeds?
        test_cases = self.generate_test_cases(behaviors, target_texts, init_embeds=None)

        # TODO: vefify that we indeed need to concatenate it,
        # and whether we need to add space or not (compare agaisnt original implementation)
        results = copy.deepcopy(conversations)
        for conv, test_case in zip(results, test_cases):
            conv[-1]["content"] += test_case

        return results


class IndivHarmBenchAttack(HarmBenchAttack):
    """
    A template method that generates test cases for a single behavior and model
    (e.g., used by GCG, PEZ, GBDA, UAT, AutoPrompt)
    """

    def __init__(
        self,
        adv_model: AdvModel,
        verbose: bool,
    ):
        """ """
        super().__init__(adv_model, verbose)

    def generate_test_cases(
        self, behaviors: list[str], targets: list[str], init_embeds: torch.Tensor | None = None
    ) -> list[str]:

        if len(behaviors) != len(targets):
            raise ValueError("The number of behaviors must match the number of targets.")

        test_cases = []
        for beh, tgt in tqdm(zip(behaviors, targets), total=len(behaviors), disable=not self.verbose):
            test_case = self.generate_test_cases_single_behavior(beh, tgt)
            test_cases.append(test_case)
        return test_cases

    @abstractmethod
    def generate_test_cases_single_behavior(self, behavior: str, target: str) -> str:
        raise NotImplementedError
