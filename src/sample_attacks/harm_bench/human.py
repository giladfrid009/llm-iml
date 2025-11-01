from src.sample_attacks.harm_bench.human_utils.manual_jailbreaks import MANUAL_JAILBREAKS
from src.sample_attacks.harm_bench.base import HarmBenchAttack
from src.adv_model import AdvModel

import random


class Human(HarmBenchAttack):
    """
    Human Jailbreaks - In the wild jailbreak presets
    [https://arxiv.org/pdf/2308.03825]
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

    def get_hparams(self) -> dict:
        return {
            "name": type(self).__name__,
            "seed": self.seed,
        }

    def generate_test_cases(self, behaviors: list[str], targets: list[str]) -> list[str]:
        jailbreaks = random.choices(MANUAL_JAILBREAKS, k=len(behaviors))
        test_cases = [jb.format(beh) for jb, beh in zip(jailbreaks, behaviors)]
        return test_cases
