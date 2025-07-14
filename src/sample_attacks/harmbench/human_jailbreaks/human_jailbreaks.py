import random

from src.sample_attacks.harmbench.human_jailbreaks.jailbreaks import JAILBREAKS
from src.sample_attacks.harmbench.baseline import HarmBenchAttack
from src.adv_model import AdvModel


class HumanJailbreaks(HarmBenchAttack):
    """
    Jailbreaks designed by humans
    """

    def __init__(
        self,
        adv_mode: AdvModel,
        seed=1,
        verbose: bool = True,
    ):
        """
        :param seed: int, random seed
        """
        super().__init__(adv_mode, verbose)
        self.seed = seed

    def generate_test_cases(self, behaviors: list[str], targets: list[str]) -> list[str]:
        jailbreaks = random.choices(JAILBREAKS, k=len(behaviors))
        test_cases = [f"{jb}\n\n{beh}" for jb, beh in zip(jailbreaks, behaviors)]
        return test_cases
