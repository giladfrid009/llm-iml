from torch import optim
import sys
import pathlib
import pandas as pd


# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from src.data import DF_Batcher
from src.eval.harmbench_evaluator import HarmBenchEvaluator
from src.eval.template_evaluator import TemplateEvaluator
from src.eval.strong_reject_evaluator import StrongRejectEvaluator
from gserve.configs import ServeConfig, LLMConfig
from src.models import load_model

from src.sample_attacks import SoftPrompt
from src.univ_attacks import UnivAttack, IML
from src.adv_model import AdvModel
from src.initialize import Initializer
from src.config import GenConfig, StopCriteria
from src.eval import Evaluator
from src.activ_extractor import ActivationExtractor


class IML_Experiment(Experiment):
    def load_data(self, train_ratio: float) -> tuple[DF_Batcher, DF_Batcher]:
        data = pd.read_csv("/home/fre.gilad/source/llm-iml/data/HarmBench/harmful_behaviors.csv")
        data = data.rename(columns={"goal": "prompt"})

        # shuffle the data
        data = data.sample(frac=1, random_state=0).reset_index(drop=True)

        split = int(train_ratio * len(data))
        ds_train = data.iloc[:split].copy()
        ds_eval = data.iloc[split:].copy()

        dl_train = DF_Batcher(ds_train, batch_size=10, shuffle=True)
        dl_eval = DF_Batcher(ds_eval, batch_size=25, shuffle=False)

        return dl_train, dl_eval

    def init_evaluators(self) -> list[Evaluator]:
        return [
            # HarmBenchEvaluator(
            #     serve_config=ServeConfig(gpu_ids=[1], startup_timeout=10 * 60, client_timeout=60),
            # ),
            StrongRejectEvaluator(
                serve_config=ServeConfig(gpu_ids=[1], startup_timeout=20 * 60, client_timeout=60),
            ),
            TemplateEvaluator(),
        ]

    def init_model(self, model_name: str) -> AdvModel:
        model, tokenizer = load_model(model_name)
        adv_model = AdvModel(model=model, tokenizer=tokenizer, num_tokens=20)
        Initializer.normal(adv_model, std=0.1)
        return adv_model

    def init_attack(self, adv_model: AdvModel, evaluators: list[Evaluator]) -> UnivAttack:
        def attack_builder(adv_model: AdvModel, epoch: int):
            return SoftPrompt(
                adv_model,
                optim_factory=lambda params: optim.AdamW(params, lr=1e-3),
                steps=15,
                mixed_precision=False,
                early_stopping=True,
            )

        optimizer = optim.Adam(adv_model.parameters(), lr=1e-2, weight_decay=0)

        activ_extractor = ActivationExtractor(
            adv_model.model,
            "lm_head",
            capture_output=True,
        )

        gen_config = GenConfig(
            max_new_tokens=512,
            do_sample=False,
            remove_invalid_values=True,
        )

        return IML(
            adv_model=adv_model,
            inner_attack=attack_builder,
            optimizer=optimizer,
            activ_extractor=activ_extractor,
            evaluators=evaluators,
            judge_metric="StrongReject/Thresh@0.5",
            eval_freq=1,
            gen_config=gen_config,
            mixed_precision=False,
            skip_already_fooled=False,
            skip_failed_attacks=True,
            dynamic_labels=20,
            log_dir="logs",
        )


if __name__ == "__main__":
    IML_Experiment().main()
