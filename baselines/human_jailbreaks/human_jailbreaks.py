from baselines.human_jailbreaks.jailbreaks import JAILBREAKS
from baselines.baseline import RedTeamingMethod
import random


class HumanJailbreaks(RedTeamingMethod):
    """
    Jailbreaks designed by humans
    """

    def __init__(self, random_subset=-1, seed=1):
        """
        :param random_subset: int, number of jailbreaks to use. If -1, use all jailbreaks
        :param seed: int, random seed
        """
        self.random_subset = random_subset
        self.seed = seed

    def generate_test_cases(self, behaviors, verbose=False, **kwargs) -> dict[str, list[str]]:
        test_cases = {}

        if type(self.random_subset) == int:
            random.seed(self.seed)

        for behavior_dict in behaviors:
            if type(self.random_subset) == int:
                random.shuffle(JAILBREAKS)
            # == behaviors metadata ==
            behavior = behavior_dict["Behavior"]
            context_str = behavior_dict["ContextString"]
            behavior_id = behavior_dict["BehaviorID"]

            if context_str:
                behavior = f"{context_str}\n\n---\n\n{behavior}"

            _test_cases = []

            if type(self.random_subset) == int:
                jailbreaks = JAILBREAKS[: self.random_subset]
            else:
                jailbreaks = JAILBREAKS

            for jb in jailbreaks:
                _test_cases.append(f"{jb}\n\n{behavior}")
            test_cases[behavior_id] = _test_cases
        return test_cases

   
