import torch
from tqdm.auto import tqdm
import copy
import random
from src.sample_attacks.base import SampleAttack, SampleOutput
from src.adv_model import AdvModel
from src.aliases import Conv
from src.utils.logging import create_logger


logger = create_logger(__name__)


# look at artifacts here: https://jailbreakbench.github.io/index#leaderboard
# ALL Harmbench artifacts here: https://github.com/justinphan3110cais/harmbench_website/tree/data/playground_data
# More artifacts here from ReNeLLM: https://drive.google.com/drive/folders/1YimdAHIDH4AEeps2MVRhTe52ucURzZMD
# EasyJailbreak results: https://github.com/EasyJailbreak/EasyJailbreak?tab=readme-ov-file#-experimental-results
# adaptive attacks artifacts: https://github.com/tml-epfl/llm-adaptive-attacks/tree/main/jailbreak_artifacts


class PA(SampleAttack):
    """
    Preset Attack: a simple preset attack that replaces input prompts with predefined adversarial prompts.
    """

    def __init__(
        self,
        adv_model: AdvModel,
        prompt_mapping: dict[str, str | list[str]],
        verbose: bool = True,
    ):
        """
        A simple preset attack that replaces input prompts with predefined adversarial prompts.

        Args:
            adv_model (AdvModel): The adversarial model to attack.
            prompt_mapping (dict[str, str | list[str]]): A mapping from original prompts to
                adversarial prompts. If the value is a list, a random prompt from the list will
                be chosen.
            verbose (bool): Whether to display a progress bar during the attack.
        """
        super().__init__(adv_model, verbose)
        self.prompt_mapping = prompt_mapping

    def get_hparams(self) -> dict:
        return {"map_size": len(self.prompt_mapping)}

    def fit(
        self,
        conversations: list[Conv],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> SampleOutput:
        conversations = copy.deepcopy(conversations)
        for conv in tqdm(conversations, disabled=not self.verbose, leave=False, desc="Attack"):
            prompt = conv[-1]["content"]
            result = self.prompt_mapping.get(prompt)

            if isinstance(result, list):
                result = random.choice(result)

            elif result is None:
                logger.warning(f"No adversarial prompt found; using original prompt: {prompt}")
                result = prompt

            conv[-1]["content"] = result

        return SampleOutput(conversations)
