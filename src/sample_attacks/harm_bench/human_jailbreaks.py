from src.sample_attacks.harm_bench.human_jailbreaks_utils.manual_jailbreaks import MANUAL_JAILBREAKS
from src.sample_attacks.harm_bench.baseline import HarmBenchAttack
from src.adv_model import AdvModel

import random


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
        jailbreaks = random.choices(MANUAL_JAILBREAKS, k=len(behaviors))
        test_cases = [jb.format(beh) for jb, beh in zip(jailbreaks, behaviors)]
        return test_cases
