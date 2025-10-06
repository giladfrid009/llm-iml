from tqdm.auto import tqdm
from abc import abstractmethod
import torch
import copy

from src.aliases import Conv
from src.adv_model import AdvModel
from src.sample_attacks.sample_attack import SampleAttack, SampleOutput


class HarmBenchAttack(SampleAttack):
    """
    A template for a red teaming method that generates test cases given a set of behaviors
    """

    def __init__(self, adv_mode: AdvModel, verbose: bool):
        self.adv_model = adv_mode
        self.verbose = verbose

        # model and tokenizer
        self.model = adv_mode.model
        self.tokenizer = adv_mode.tokenizer

    def fit(
        self,
        conversations: list[Conv],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> SampleOutput:
        if len(conversations) != len(target_texts):
            raise ValueError("The number of conversations must match the number of target texts.")

        assert all(len(conv) == 1 for conv in conversations)
        assert all(conv[-1]["role"] == "user" for conv in conversations)

        behaviors = [conv[-1]["content"] for conv in conversations]
        test_cases = self.generate_test_cases(behaviors, target_texts)

        adv_convs = copy.deepcopy(conversations)
        for conv, test_case in zip(adv_convs, test_cases):
            conv[-1]["content"] = test_case

        return SampleOutput(adv_convs)

    @abstractmethod
    def generate_test_cases(self, behaviors: list[str], targets: list[str]) -> list[str]:
        raise NotImplementedError


class SequentialHarmBenchAttack(HarmBenchAttack):
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

    def generate_test_cases(self, behaviors: list[str], targets: list[str]) -> list[str]:
        if len(behaviors) != len(targets):
            raise ValueError("The number of behaviors must match the number of targets.")

        test_cases = []
        for beh, tgt in tqdm(zip(behaviors, targets), total=len(behaviors), disable=not self.verbose, leave=False):
            test_case = self.generate_test_cases_single_behavior(beh, tgt)
            test_cases.append(test_case)
        return test_cases

    @abstractmethod
    def generate_test_cases_single_behavior(self, behavior: str, target: str) -> str:
        raise NotImplementedError
