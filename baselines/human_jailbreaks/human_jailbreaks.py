from baselines.human_jailbreaks.jailbreaks import JAILBREAKS
from baselines.baseline import RedTeamingMethod
import random


class HumanJailbreaks(RedTeamingMethod):
    """
    Jailbreaks designed by humans
    """

    def __init__(self, seed=1):
        """
        :param seed: int, random seed
        """
        self.seed = seed

    def generate_test_cases(self, behaviors: list[str], targets: list[str]) -> list[str]:
        jailbreaks = random.choices(JAILBREAKS, k=len(behaviors))
        test_cases = [f"{jb}\n\n{beh}" for jb, beh in zip(jailbreaks, behaviors)]
        return test_cases
