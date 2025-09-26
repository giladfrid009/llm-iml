import sys
import pathlib


# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from src.eval.harmbench_evaluator import HarmBenchEvaluator
from src.eval.template_evaluator import TemplateEvaluator
from src.eval.strong_reject_evaluator import StrongRejectEvaluator
from gserve.configs import ServeConfig, LLMConfig
from src.models import load_model

from src.univ_attacks import UnivAttack, UnivSoftPrompt
from src.fgsm_optim import FGSM
from src.adv_model import AdvModel
from src.initialize import Initializer
from src.config import GenConfig, StopCriteria
from src.eval import Evaluator


class SoftPrompt_Experiment(Experiment):
    def init_evaluators(self) -> list[Evaluator]:
        return [
            # HarmBenchEvaluator(
            #     serve_config=ServeConfig(
            #         gpu_ids=[1],
            #         startup_timeout=10 * 60,
            #         client_timeout=60,
            #         verbose=False,
            #     ),
            # ),
            StrongRejectEvaluator(
                serve_config=ServeConfig(gpu_ids=[1], startup_timeout=20 * 60, client_timeout=60),
            ),
            TemplateEvaluator(),
        ]

    def init_model(self, model_name: str) -> AdvModel:
        model, tokenizer = load_model(model_name)
        adv_model = AdvModel(model=model, tokenizer=tokenizer, num_tokens=20)
        Initializer.from_string(adv_model, "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !", strict=False)
        return adv_model

    def init_attack(self, adv_model: AdvModel, evaluators: list[Evaluator]) -> UnivAttack:
        gen_config = GenConfig(
            max_new_tokens=512,
            do_sample=False,
            remove_invalid_values=True,
        )

        optimizer = FGSM(
            adv_model.parameters(),
            lr=0.001,
        )

        return UnivSoftPrompt(
            adv_model=adv_model,
            optimizer=optimizer,
            evaluators=evaluators,
            judge_metric="StrongReject/Thresh@0.5",
            eval_freq=10,
            gen_config=gen_config,
            mixed_precision=False,
            log_dir="logs",
        )


if __name__ == "__main__":
    SoftPrompt_Experiment().main()
