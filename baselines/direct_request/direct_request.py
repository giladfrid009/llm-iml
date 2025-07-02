from baselines.baseline import RedTeamingMethod


class DirectRequest(RedTeamingMethod):
    def __init__(self, **kwargs):
        pass

    def generate_test_cases(self, behaviors, verbose=False):
        """
        Generates test cases for the provided behaviors by simply returning the behavior string as the test case.
        Since the behavior strings are formatted as commands, they can be directly used as test cases.

        :param behaviors: a list of behavior dictionaries
        :param verbose: a boolean indicating whether to print verbose output
        :return: a dictionary of test cases and None for the logs
        """
        test_cases = {}
        for behavior_dict in behaviors:
            behavior = behavior_dict["Behavior"]
            context_str = behavior_dict["ContextString"]
            behavior_id = behavior_dict["BehaviorID"]
            if context_str:
                behavior = f"{context_str}\n\n---\n\n{behavior}"
            test_cases[behavior_id] = [behavior]
        return test_cases, None

    