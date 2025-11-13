from argparse import ArgumentParser
from torch import optim
import sys
import pathlib
import pandas as pd

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.run_iml import IML_Experiment
from src.sample_attacks import PCAV, LogisticTrainer
from src.univ_attacks import UnivAttack, IML
from src.adv_model import AdvModel
from src.activ_extractor import ActivationExtractor
from src.data import TableLoader
from src.utils.logging import create_logger


logger = create_logger(__name__)


class IML_PCAV_Experiment(IML_Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        super().add_arguments(parser)

        parser.set_defaults(
            model="meta-llama/Llama-2-7b-chat-hf",
            lr=1e-2,
            skip_fooled="true",
            skip_failed="true",
            dynamic_labels=20,
            warmup_epochs=4,
            layer_names=["model.layers.12", "model.layers.17", "model.layers.25", "lm_head"],
        )

    def load_data(self):
        scav_data = pd.read_json("data/scav/original.json")
        scav_train = scav_data[scav_data["split"] == "train"]
        scav_eval = scav_data[scav_data["split"] == "test"]
        return scav_train, scav_eval

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
        cav_train, cav_eval = self.load_data()
        dl_train = TableLoader(cav_train, batch_size=50, shuffle=False)
        dl_eval = TableLoader(cav_eval, batch_size=50, shuffle=False)

        logger.info("Training CAV Classifiers...")
        cav_extractor = ActivationExtractor(adv_model.model, "lm_head", capture_output=False)
        logistic_trainer = LogisticTrainer(adv_model, cav_extractor)
        classifiers = logistic_trainer.fit(dl_train, dl_eval)

        inner_attack = PCAV(
            adv_model,
            classifiers=classifiers,
            activ_extractor=cav_extractor,
            optim_factory=lambda params: optim.Adam(params, lr=5e-3),
            steps=20,
            min_acc=0.9,
            target_prob=0.05,
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
    IML_PCAV_Experiment().main()
