from baselines.baseline import RedTeamingMethod


class DirectRequest(RedTeamingMethod):
    def __init__(self, **kwargs):
        pass

    def generate_test_cases(self, behaviors: list[str], targets: list[str]) -> list[str]:
        return behaviors

    