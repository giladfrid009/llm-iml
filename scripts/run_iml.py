from argparse import ArgumentParser
from torch import optim
import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from src.sample_attacks import SP
from src.univ_attacks import UnivAttack, IML
from src.adv_model import AdvModel
from src.initialize import Initializer
from src.activ_extractor import ActivationExtractor


class IML_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        iml_args = parser.add_argument_group("IML Attack Parameters")

        iml_args.add_argument(
            "--num_tokens",
            type=int,
            default=20,
            metavar="NUM",
            help="Number of tokens in the adversarial trigger.",
        )

        iml_args.add_argument(
            "--lr",
            type=float,
            default=1e-2,
            metavar="FLOAT",
            help="Learning rate for the adversarial trigger optimization.",
        )

        iml_args.add_argument(
            "--layers",
            type=str,
            nargs="+",
            default=["lm_head"],
            metavar="NAMES",
            help="Names of the layers to extract activations from.",
        )

        iml_args.add_argument(
            "--skip_fooled",
            type=str,
            default="true",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to skip samples that are already fooled by the universal trigger.",
        )

        iml_args.add_argument(
            "--skip_failed",
            type=str,
            default="true",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to skip samples that the inner attack failed to attack.",
        )

        iml_args.add_argument(
            "--dynamic_labels",
            type=int,
            default=40,
            metavar="N",
            help="Number of dynamic labels to use for each sample during IML training. If 0 no dynamic labels are used.",
        )

        iml_args.add_argument(
            "--warmup_epochs",
            type=int,
            default=2,
            metavar="NUM",
            help="Number of warmup epochs.",
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
        metric_logger,
    ) -> UnivAttack:
        args = self.args()

        inner_attack = SP(
            adv_model,
            optim_factory=lambda params: optim.AdamW(params, lr=1e-2),
            steps=25,
            target_matching=True,
        )

        optimizer = optim.Adam(
            adv_model.parameters(),
            lr=args.lr,
            weight_decay=0,
        )

        activ_extractor = ActivationExtractor(
            adv_model.model,
            *args.layers,
            capture_output=False,
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
            skip_already_fooled=args.skip_fooled == "true",
            skip_failed_attacks=args.skip_failed == "true",
            warmup_epochs=args.warmup_epochs,
            dynamic_labels=args.dynamic_labels,
        )


if __name__ == "__main__":
    IML_Experiment().main()
