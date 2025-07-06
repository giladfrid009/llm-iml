from tqdm.auto import tqdm
import numpy as np
import transformers
import vllm

from src.adver_model import AdverModel


class RedTeamingMethod:
    default_dependencies = [transformers, vllm]

    """
    A template for a red teaming method that generates test cases given a set of behaviors
    """

    def __init__(self, adv_mode: AdverModel, verbose: bool):
        self.adv_model = adv_mode
        self.verbose = verbose

        # model and tokenizer
        self.model = adv_mode.model
        self.tokenizer = adv_mode.tokenizer

    def generate_test_cases(self, behaviors: list[str], targets: list[str]) -> list[str]:
        raise NotImplementedError


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

    def generate_test_cases(self, behaviors: list[str], targets: list[str]) -> list[str]:

        if len(behaviors) != len(targets):
            raise ValueError("The number of behaviors must match the number of targets.")

        test_cases = []
        for beh, tgt in tqdm(zip(behaviors, targets), disable=not self.verbose):
            test_case = self.generate_test_cases_single_behavior(beh, tgt)
            test_cases.append(test_case)
        return test_cases

    def generate_test_cases_single_behavior(self, behavior: str, target: str) -> str:
        raise NotImplementedError
