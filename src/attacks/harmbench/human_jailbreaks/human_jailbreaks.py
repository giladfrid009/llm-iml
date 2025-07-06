from src.attacks.harmbench.human_jailbreaks.jailbreaks import JAILBREAKS
from src.attacks.harmbench.baseline import RedTeamingMethod
import random
import torch

class HumanJailbreaks(RedTeamingMethod):
    """
    Jailbreaks designed by humans
    """

    def __init__(self, seed=1):
        """
        :param seed: int, random seed
        """
        self.seed = seed

    def generate_test_cases(self, behaviors: list[str], targets: list[str], init_embeds: torch.Tensor | None = None) -> list[str]:
        jailbreaks = random.choices(JAILBREAKS, k=len(behaviors))
        test_cases = [f"{jb}\n\n{beh}" for jb, beh in zip(jailbreaks, behaviors)]
        return test_cases
