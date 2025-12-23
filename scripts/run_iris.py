from argparse import ArgumentParser
import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from src.univ_attacks import UnivAttack, IRIS
from src.univ_attacks.iris import RefusalConfig
from src.fgsm_optim import FGSM
from src.adv_model import AdvModel
from src.initialize import Initializer


class IRIS_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        iris_group = parser.add_argument_group("IRIS + Soft-Prompt Attack Parameters")

        iris_group.add_argument(
            "--num_tokens",
            type=int,
            default=20,
            metavar="NUM",
            help="Number of tokens in the adversarial trigger.",
        )

        iris_group.add_argument(
            "--lr",
            type=float,
            metavar="FLOAT",
            default=0.001,
            help="Learning rate for FGSM optimizer.",
        )

        iris_group.add_argument(
            "--refusal_config",
            type=str,
            metavar="PATH",
            required=True,
            help="Path to the refusal configuration file, of type RefusalConfig.",
        )

        iris_group.add_argument(
            "--beta",
            type=float,
            metavar="FLOAT",
            default=0.5,
            help="Balance factor between refusal and target loss.",
        )

        parser.set_defaults(
            train_batch=16,
            max_epochs=2000,
        )

    def create_adversarial_model(self, model, tokenizer) -> AdvModel:
        adv_model = AdvModel(model, tokenizer, num_tokens=self.args().num_tokens, add_spaces=False, adv_suffix=True)
        embeds = Initializer.random_normal(adv_model, std=0.1)
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
        metric_tracker,
    ) -> UnivAttack:
        """
        Creates a IRIS universal adversarial attack instance.
        """
        args = self.args()
        optimizer = FGSM(adv_model.parameters(), lr=args.lr)
        refusal_config = RefusalConfig.load(args.refusal_config)

        return IRIS(
            adv_model=adv_model,
            optimizer=optimizer,
            refusal_config=refusal_config,
            beta=args.beta,
            evaluators=evaluators,
            eval_metric=eval_metric,
            eval_freq=eval_freq,
            gen_config=gen_config,
            mixed_precision=mixed_precision,
            metric_tracker=metric_tracker,
        )


if __name__ == "__main__":
    IRIS_Experiment().main()
