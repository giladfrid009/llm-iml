from argparse import ArgumentParser
import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from src.univ_attacks import UnivAttack, SoftPrompt
from src.fgsm_optim import FGSM
from src.adv_model import AdvModel
from src.initialize import Initializer


class SoftPrompt_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        soft_group = parser.add_argument_group("Soft-Prompt Attack Parameters")

        soft_group.add_argument(
            "--lr",
            type=float,
            metavar="FLOAT",
            default=0.001,
            help="Learning rate for FGSM optimizer.",
        )

        soft_group.add_argument(
            "--init_str",
            type=str,
            metavar="STR",
            default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
            help="Initialization string for the soft prompt embeddings.",
        )

        parser.set_defaults(
            train_batch=16,
            max_epochs=2000,
            max_time=60,
            eval_freq=10,
        )

    def create_adversarial_model(self, model, tokenizer) -> AdvModel:
        adv_model = AdvModel(model, tokenizer, num_tokens=20, add_spaces=False, adv_suffix=True)
        embeds = Initializer.from_string(adv_model, self.args().init_str, strict=False)
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
        """
        Creates a soft-prompt universal adversarial attack instance.
        """

        optimizer = FGSM(
            adv_model.parameters(),
            lr=self.args().lr,
        )

        return SoftPrompt(
            adv_model=adv_model,
            optimizer=optimizer,
            evaluators=evaluators,
            eval_metric=eval_metric,
            eval_freq=eval_freq,
            gen_config=gen_config,
            mixed_precision=mixed_precision,
            metric_logger=metric_logger,
        )


if __name__ == "__main__":
    SoftPrompt_Experiment().main()
