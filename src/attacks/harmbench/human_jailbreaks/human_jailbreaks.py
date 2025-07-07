from src.attacks.harmbench.human_jailbreaks.jailbreaks import JAILBREAKS
from src.attacks.harmbench.baseline import RedTeamingMethod
from src.adver_model import AdverModel
import random
import torch

class HumanJailbreaks(RedTeamingMethod):
    """
    Jailbreaks designed by humans
    """

    def __init__(self, adv_mode: AdverModel, seed=1, verbose: bool = True, ):
        """
        :param seed: int, random seed
        """
        super().__init__(adv_mode, verbose)
        self.seed = seed

    def generate_test_cases(self, behaviors: list[str], targets: list[str], init_embeds: torch.Tensor | None = None) -> list[str]:
        jailbreaks = random.choices(JAILBREAKS, k=len(behaviors))
        test_cases = [f"{jb}\n\n{beh}" for jb, beh in zip(jailbreaks, behaviors)]
        return test_cases
