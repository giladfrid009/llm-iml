from argparse import ArgumentParser
import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).resolve().parent.parent.parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.run_iml import IML_Experiment


class Exp(IML_Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        super().add_arguments(parser)

        parser.set_defaults(
            model="meta-llama/Llama-2-7b-chat-hf",
            layers=["model.layers.12", "model.layers.17", "model.layers.25", "lm_head"],
            lr=1e-2,
            skip_fooled="true",
            skip_failed="true",
            dynamic_labels=40,
            warmup_epochs=2,
            target_controls="false",
            # inner-attack params
            warmup_attack_lr=5e-3,
            warmup_attack_steps=45,
            warmup_attack_target_matching="false",
            main_attack_lr=5e-3,
            main_attack_steps=15,
            main_attack_target_matching="false",
        )


if __name__ == "__main__":
    Exp().main()
