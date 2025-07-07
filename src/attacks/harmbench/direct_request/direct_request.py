from src.attacks.harmbench.baseline import RedTeamingMethod
from src.adver_model import AdverModel
import torch

class DirectRequest(RedTeamingMethod):
    def __init__(self, adv_mode: AdverModel, verbose: bool = True):
        super().__init__(adv_mode, verbose)

    def generate_test_cases(self, behaviors: list[str], targets: list[str], init_embeds: torch.Tensor | None = None) -> list[str]:
        return behaviors

    