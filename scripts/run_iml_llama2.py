from argparse import ArgumentParser
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
from src.activ_extractor import ActivationExtractor
from src.fgsm_optim import FGSM


class IML_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.set_defaults(
            model="meta-llama/Llama-2-7b-chat-hf",
        )

    def create_adversarial_model(self, model, tokenizer) -> AdvModel:
        adv_model = AdvModel(model, tokenizer, num_tokens=20, add_spaces=False, adv_suffix=True)
        embeds = Initializer.random_normal(adv_model, std=0.1)  # High STD = Worse
        adv_model.set_embeddings(embeds)
        return adv_model

    def initialize_attack(
        self,
        adv_model: AdvModel,
        evaluators,
        eval_metric,
        eval_freq,
        mixed_precision,
        gen_config,
        metric_logger,
    ) -> UnivAttack:
        inner_attack = SoftPrompt(
            adv_model,
            optim_factory=lambda params: optim.AdamW(params, lr=5e-3),
            steps=15,
            mixed_precision=False,
            early_stopping=True,
        )

        optimizer = optim.Adam(
            adv_model.parameters(),
            lr=1e-2,
            weight_decay=0,
        )

        activ_extractor = ActivationExtractor(
            adv_model.model,
            # "model.layers.12",
            # "model.layers.17",
            # "model.layers.25",
            "lm_head",
            capture_output=True,
        )

        return IML(
            adv_model=adv_model,
            inner_attack=inner_attack,
            optimizer=optimizer,
            activ_extractor=activ_extractor,
            evaluators=evaluators,
            eval_metric=eval_metric,
            eval_freq=eval_freq,
            gen_config=gen_config,
            mixed_precision=mixed_precision,
            metric_logger=metric_logger,
            # specialized args
            skip_already_fooled=False,
            skip_failed_attacks=True,
            dynamic_labels=20,
        )


if __name__ == "__main__":
    IML_Experiment().main()
