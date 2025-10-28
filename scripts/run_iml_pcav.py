from argparse import ArgumentParser
from torch import optim
import sys
import pathlib
import pandas as pd

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from src.sample_attacks import PCAV, LogisticTrainer
from src.univ_attacks import UnivAttack, IML
from src.adv_model import AdvModel
from src.initialize import Initializer
from src.activ_extractor import ActivationExtractor
from src.data import TableLoader
from src.utils.logging import create_logger


logger = create_logger(__name__)


class IML_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.set_defaults(
            model="meta-llama/Llama-2-7b-chat-hf",
        )

    def load_data(self):
        scav_data = pd.read_json("data/scav/original.json")
        scav_train = scav_data[scav_data["split"] == "train"]
        scav_eval = scav_data[scav_data["split"] == "test"]
        return scav_train, scav_eval

    def create_adversarial_model(self, model, tokenizer) -> AdvModel:
        adv_model = AdvModel(model, tokenizer, num_tokens=20, add_spaces=False, adv_suffix=True)
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

        optimizer = optim.Adam(
            adv_model.parameters(),
            lr=1e-2,
            weight_decay=0,
        )

        activ_extractor = ActivationExtractor(
            adv_model.model,
            "model.layers.12",
            "model.layers.17",
            "model.layers.25",
            "lm_head",
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
            skip_already_fooled=True,
            skip_failed_attacks=True,
            dynamic_labels=20,
        )


if __name__ == "__main__":
    IML_Experiment().main()
