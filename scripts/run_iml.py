from torch import optim
import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from src.sample_attacks import SoftPrompt
from src.univ_attacks import UnivAttack, IML
from src.adv_model import AdvModel
from src.initialize import Initializer
from src.config import GenConfig, StopCriteria
from src.activ_extractor import ActivationExtractor


class IML_Experiment(Experiment):
    def create_adversarial_model(self, model, tokenizer) -> AdvModel:
        # TODO: try less tokens
        # TODO: try different initializations
        adv_model = AdvModel(model=model, tokenizer=tokenizer, num_tokens=20)
        Initializer.normal(adv_model, std=0.1)  # High STD = Worse
        # Initializer.from_string(adv_model, "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !", strict=False) # USUALLY PERFORMS WORSE
        return adv_model

    def initialize_attack(self, adv_model, evaluators, metric_logger) -> UnivAttack:
        gen_config = GenConfig(
            max_new_tokens=512,
            do_sample=True,
            # remove_invalid_values=True,
            # top_p=0.9,
            # temperature=0.6,
        )

        # TODO: try Adam - doesnt do much difference, maybe worse
        # TODO: we can create an attack_builder func and try with inner_attack scheduling,
        # i.e. scheduling the number of steps
        # TODO: try with early_stopping=False
        inner_attack = SoftPrompt(
            adv_model,
            optim_factory=lambda params: optim.AdamW(params, lr=1e-2),
            steps=25,
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
        # try also internal layer: i.e lm_head, capture_output=False:
        # - on regular Llama2 performs worse
        # - on GraySwanAI/Llama-3-8B-Instruct-RR performs 2x better
        # TODO: try combination of output layer + internal layer
        activ_extractor = ActivationExtractor(
            adv_model.model,
            "lm_head",
            "model.layers.17",
            "model.layers.12",
            "model.layers.25",
            capture_output=True,
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
            # judge_metric="LlamaGuard/Llama-Guard-3-8B",
            eval_freq=2,
            gen_config=gen_config,
            mixed_precision=False,
            skip_already_fooled=False,
            skip_failed_attacks=True,
            dynamic_labels=40,
            metric_logger=metric_logger,
        )


if __name__ == "__main__":
    IML_Experiment().main()
