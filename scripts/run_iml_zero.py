from argparse import ArgumentParser
from torch import optim
import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from src.sample_attacks import SoftPromptZero
from src.univ_attacks import UnivAttack, IML
from src.adv_model import AdvModel
from src.initialize import Initializer
from src.config import GenConfig, StopCriteria
from src.activ_extractor import ActivationExtractor


class IMLZero_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        pass
    
    def create_adversarial_model(self, model, tokenizer) -> AdvModel:
        adv_model = AdvModel(model=model, tokenizer=tokenizer, num_tokens=20)
        Initializer.normal(adv_model, std=0.1)
        return adv_model

    def initialize_attack(self, adv_model, evaluators, metric_logger) -> UnivAttack:
        gen_config = GenConfig(
            max_new_tokens=512,
            do_sample=True,
            remove_invalid_values=True,
        )

        inner_attack = SoftPromptZero(
            adv_model,
            optim_factory=lambda params: optim.AdamW(params, lr=1e-2),
            inject_func=lambda a, x: a.inject_tokens(x, add_spaces=False, adv_suffix=False),
            steps=25,
            mixed_precision=False,
            early_stopping=True,
        )

        optimizer = optim.Adam(
            adv_model.parameters(),
            lr=5e-3,
            weight_decay=0,
        )

        activ_extractor = ActivationExtractor(
            adv_model.model,
            "lm_head",
            capture_output=False,
        )

        return IML(
            adv_model=adv_model,
            inner_attack=inner_attack,
            optimizer=optimizer,
            activ_extractor=activ_extractor,
            evaluators=evaluators,
            eval_metric="StrongReject/Thresh@0.5",
            # eval_metric="LlamaGuard/Meta-Llama-Guard-2-8B",
            eval_freq=0.5,
            gen_config=gen_config,
            mixed_precision=False,
            skip_already_fooled=False,
            skip_failed_attacks=True,
            dynamic_labels=20,
            metric_logger=metric_logger,
        )


if __name__ == "__main__":
    IMLZero_Experiment().main()
