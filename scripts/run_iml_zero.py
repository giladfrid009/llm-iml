from argparse import ArgumentParser
from torch import optim
import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.run_iml import IML_Experiment
from src.sample_attacks import SPZ
from src.univ_attacks import UnivAttack, IML
from src.adv_model import AdvModel
from src.activ_extractor import ActivationExtractor


class IML_Zero_Experiment(IML_Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        super().add_arguments(parser)

        parser.set_defaults(
            model="meta-llama/Llama-2-7b-chat-hf",
            layers=["lm_head"],
            lr=5e-3,
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
        inner_attack = SPZ(
            adv_model,
            optim_factory=lambda params: optim.AdamW(params, lr=1e-2),
            inject_func=lambda a, x: a.inject_tokens(x, add_spaces=False, adv_suffix=False),
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
    IML_Zero_Experiment().main()
