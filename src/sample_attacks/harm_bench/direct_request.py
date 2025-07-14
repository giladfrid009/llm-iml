from src.sample_attacks.harm_bench.baseline import HarmBenchAttack
from src.adv_model import AdvModel


class DirectRequest(HarmBenchAttack):
    def __init__(self, adv_mode: AdvModel, verbose: bool = True):
        super().__init__(adv_mode, verbose)

    def generate_test_cases(self, behaviors: list[str], targets: list[str]) -> list[str]:
        return behaviors
