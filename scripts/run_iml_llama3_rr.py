from argparse import ArgumentParser
from torch import optim
import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.run_iml import IML_Experiment
from src.sample_attacks import SP
from src.univ_attacks import UnivAttack, IML
from src.adv_model import AdvModel
from src.activ_extractor import ActivationExtractor


class IML_Llama3RR_Experiment(IML_Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        super().add_arguments(parser)

        parser.set_defaults(
            model="GraySwanAI/Llama-3-8B-Instruct-RR",
            layers=["model.layers.12", "model.layers.17", "model.layers.25", "lm_head"],
            lr=1e-2,
            skip_fooled="true",
            skip_failed="true",
            dynamic_labels=20,
            warmup_epochs=4,
        )

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
        def sample_attack_factory(adv_model: AdvModel, epoch: int):
            if epoch < args.warmup_epochs:
                return SP(
                    adv_model,
                    optim_factory=lambda params: optim.AdamW(params, lr=1e-2),
                    steps=65,
                    target_matching=False,
                )

            return SP(
                adv_model,
                optim_factory=lambda params: optim.AdamW(params, lr=1e-2),
                steps=25,
                target_matching=True,
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

        return IML(
            adv_model=adv_model,
            inner_attack=sample_attack_factory,
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
    IML_Llama3RR_Experiment().main()
