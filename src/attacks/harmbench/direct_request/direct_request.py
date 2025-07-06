from src.attacks.harmbench.baseline import RedTeamingMethod
import torch

class DirectRequest(RedTeamingMethod):
    def __init__(self, **kwargs):
        pass

    def generate_test_cases(self, behaviors: list[str], targets: list[str], init_embeds: torch.Tensor | None = None) -> list[str]:
        return behaviors

    