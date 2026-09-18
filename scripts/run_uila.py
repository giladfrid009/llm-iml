from argparse import ArgumentParser
from torch import optim
import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.run_upd import UPD_Experiment
from src.sample_attacks import SP
from src.univ_attacks import UnivAttack
from src.univ_attacks.uila import UILA
from src.adv_model import AdvModel
from src.activ_extractor import ActivationExtractor


class UILA_Experiment(UPD_Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        super().add_arguments(parser)

        parser.add_argument(
            "--normalized_loss",
            type=str,
            default="true",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to use normalized loss in ILA.",
        )

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
        def sample_attack_factory(adv_model: AdvModel, position):
            if position.epoch < args.warmup_epochs:
                return SP(
                    adv_model,
                    optim_factory=lambda params: optim.AdamW(params, lr=args.warmup_attack_lr),
                    steps=args.warmup_attack_steps,
                    target_matching=args.warmup_attack_target_matching == "true",
                    kv_caching=args.kv_caching == "true",
                )

            return SP(
                adv_model,
                optim_factory=lambda params: optim.AdamW(params, lr=args.main_attack_lr),
                steps=args.main_attack_steps,
                target_matching=args.main_attack_target_matching == "true",
                kv_caching=args.kv_caching == "true",
            )

        args = self.args()

        optimizer = optim.Adam(
            adv_model.parameters(),
            lr=args.lr,
        )

        activ_extractor = ActivationExtractor(
            adv_model.model,
            *args.layers,
            capture_output=False,
        )

        return UILA(
            adv_model=adv_model,
            inner_attack=sample_attack_factory,
            optimizer=optimizer,
            activ_extractor=activ_extractor,
            evaluators=evaluators,
            eval_metric=eval_metric,
            eval_freq=eval_freq,
            gen_config=gen_config,
            mixed_precision=mixed_precision,
            metric_tracker=metric_tracker,
            # specialized args
            skip_already_fooled=args.skip_fooled == "true",
            skip_failed_attacks=args.skip_failed == "true",
            target_controls=args.target_controls == "true",
            warmup_epochs=args.warmup_epochs,
            dynamic_labels=args.dynamic_labels,
            normalized_loss=args.normalized_loss == "true",
        )


if __name__ == "__main__":
    UILA_Experiment().main()
