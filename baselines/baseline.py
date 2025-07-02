# ============================== BASELINE TEMPLATE ============================== #
import time
from tqdm.auto import tqdm
import numpy as np
import transformers
import vllm

from src.adver_model import AdverModel


class RedTeamingMethod:
    default_dependencies = [transformers, vllm]

    """
    A template for a red teaming method that generates test cases given a set of behaviors
    """

    def __init__(self):
        raise NotImplementedError

    def generate_test_cases(self, behaviors: list[dict[str, str]], verbose: bool = False) -> dict[str, list[str]]:
        """
        Generates test cases for the provided behaviors.
        For text behaviors, test cases are strings.
        For multimodal behaviors, test cases are lists of (image_filename, string).
        (For portability, the path to image files is provided separately while generating completions.)

        :param behaviors: a list of behavior dictionaries specifying the behaviors to generate test cases for
        :param verbose: whether to print progress
        :return: (test_cases, logs), where both are dictionaries with behavior IDs as keys
        """
        raise NotImplementedError


class SingleBehaviorRedTeamingMethod(RedTeamingMethod):
    """
    A template method that generates test cases for a single behavior and model
    (e.g., used by GCG, PEZ, GBDA, UAT, AutoPrompt)
    """

    def __init__(
        self,
        adv_model: AdverModel,
        num_test_cases_per_behavior: int = 1,
        test_cases_batch_size: int = 1,
    ):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param num_test_cases_per_behavior: the number of test cases to generate in a call to self.generate_test_cases
        :param test_cases_batch_size: the batch size to use for calling num_generate (for dividing up num_test_cases_per_behavior)
        """
        model, tokenizer = adv_model.model, adv_model.tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.num_test_cases_per_behavior = num_test_cases_per_behavior
        self.test_cases_batch_size = test_cases_batch_size

    def generate_test_cases(self, behaviors: list[dict[str, str]], verbose: bool = False) -> dict[str, list[str]]:
        """
        Generates test cases for the provided behaviors. The outputs of this method are passed to the
        save_test_cases method, which saves test cases and logs to disk.

        :param behaviors: a list of behavior dictionaries specifying the behaviors to generate test cases for
        :param verbose: whether to print progress
        :return: a dictionary of test cases, where the keys are the behavior IDs and the values are lists of test cases
        """
        # set self.test_cases_batch_size and self.num_test_cases_per_behavior to 1 if not defined yet
        # (enables subclassing with a new __init__ function that doesn't define these attributes)
        if not hasattr(self, "test_cases_batch_size"):
            self.test_cases_batch_size = 1
        if not hasattr(self, "num_test_cases_per_behavior"):
            self.num_test_cases_per_behavior = 1

        test_cases = {}
        for behavior_dict in tqdm(behaviors, total=len(behaviors), disable=not verbose):
            start_time = time.time()
            # break it down into batches
            num_batches = int(np.ceil(self.num_test_cases_per_behavior / self.test_cases_batch_size))
            remainder = self.num_test_cases_per_behavior % self.test_cases_batch_size
            current_test_cases = []
            
            for j in tqdm(range(num_batches), disable=not verbose, desc="Attack"):
                # Calculate the current batch size
                if j == num_batches - 1 and remainder != 0:
                    current_batch_size = remainder
                else:
                    current_batch_size = self.test_cases_batch_size

                batch_test_cases = self.generate_test_cases_single_behavior(
                    behavior_dict=behavior_dict,
                    num_generate=current_batch_size,
                    verbose=verbose,
                )

                if isinstance(batch_test_cases, list):
                    current_test_cases.extend(batch_test_cases)
                else:
                    current_test_cases.append(batch_test_cases)

            # add test cases and info to the output dictionaries
            behavior_id = behavior_dict["BehaviorID"]

            test_cases[behavior_id] = current_test_cases

        return test_cases

    def generate_test_cases_single_behavior(self, behavior_dict: dict[str, str], num_generate: int, verbose: bool) -> list[str] | str:
        """
        Generates test cases for a specific behavior. This is the main method that should be implemented by subclasses.

        This method can return a single test case and logs, or a list of test cases and logs. We check the type of the
        output and handle it accordingly in the `generate_test_cases` method. Note that this can cause issues if an
        individual test case and individual log are both lists already and have the same length. This can only happen
        for multimodal attacks, so it is recommended to wrap test cases and logs in lists for multimodal attacks.

        :param behavior_dict: a behavior dictionary specifying the behavior to generate test cases for
        :param num_generate: the number of test cases to generate (for attacks that can generate multiple test cases at once)
        :param verbose: whether to print progress
        :return: list(test_cases) OR test_case; a list of test cases or a single test.
        """
        raise NotImplementedError
