from torch import optim
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

from src.sample_attacks import SoftPrompt
from src.univ_attacks import UnivAttack, IML
from src.adv_model import AdvModel
from src.initialize import Initializer
from src.config import GenConfig, StopCriteria
from src.eval import Evaluator
from src.activ_extractor import ActivationExtractor


class IML_Experiment(Experiment):
    def init_evaluators(self) -> list[Evaluator]:
        return [
            # HarmBenchEvaluator(
            #     serve_config=ServeConfig(gpu_ids=[1], startup_timeout=10 * 60, client_timeout=60),
            # ),
            StrongRejectEvaluator(
                serve_config=ServeConfig(gpu_ids=[1], startup_timeout=20 * 60, client_timeout=60),
            ),
            TemplateEvaluator(),
        ]

    def init_model(self, model_name: str) -> AdvModel:
        model, tokenizer = load_model(model_name)
        
        # TODO: try less tokens
        # TODO: try different initializations
        adv_model = AdvModel(model=model, tokenizer=tokenizer, num_tokens=20)
        Initializer.normal(adv_model, std=0.1) # USUALLY PERFORMS BETTER
        # Initializer.from_string(adv_model, "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !", strict=False) # USUALLY PERFORMS WORSE
        return adv_model

    def init_attack(self, adv_model: AdvModel, evaluators: list[Evaluator]) -> UnivAttack:
        # TODO: try Adam - doesnt do much difference, maybe worse
        # TODO: we can create an attack_builder func and try with inner_attack scheduling,
        # i.e. scheduling the number of steps
        # TODO: try with early_stopping=False
        inner_attack = SoftPrompt(
            adv_model,
            optim_factory=lambda params: optim.Adam(params, lr=1e-3),
            steps=15,
            mixed_precision=False,
            early_stopping=True,
        )

        # TODO: try different optimizers maybe FGSM and AdamW
        # (AdamW probably significantly worse by previous experiments)
        optimizer = optim.Adam(
            adv_model.parameters(),
            lr=1e-2,
            weight_decay=0,
        )

        # TODO: lm_head is the last layer so we basically optimize over the logits
        # try also internal layer: i.e lm_head, capture_output=False - PERFORMS WORSE
        # TODO: try combination of output layer + internal layer
        activ_extractor = ActivationExtractor(
            adv_model.model,
            "lm_head",
            capture_output=True,
        )

        gen_config = GenConfig(
            max_new_tokens=512,
            do_sample=False,
            remove_invalid_values=True,
        )

        # TODO: try without dynamic labels and different amount
        # TODO: try with skip_already_fooled=True - doesnt do much difference, maybe worse
        # TODO: try with skip_failed_attacks=False (for ablations)
        # TODO: try with mixed_precision=True for IML and inner_attack, it changes the results - NOT NEEDED SINCE MODEL TYPE IS BF16, WHICH HAS SAME RANGE AS FP32
        return IML(
            adv_model=adv_model,
            inner_attack=inner_attack,
            optimizer=optimizer,
            activ_extractor=activ_extractor,
            evaluators=evaluators,
            judge_metric="StrongReject/Thresh@0.5",
            eval_freq=1,
            gen_config=gen_config,
            mixed_precision=False,
            skip_already_fooled=False,
            skip_failed_attacks=True,
            dynamic_labels=20,
            log_dir="logs",
        )


if __name__ == "__main__":
    IML_Experiment().main()
