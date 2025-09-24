import torch
from torch import optim
import sys
import pathlib
import pandas as pd
import random

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.utils import env
from src.utils.logging import create_logger
from src.data import DF_Batcher
from src.eval.harmbench_evaluator import HarmBenchEvaluator
from src.eval.template_evaluator import TemplateEvaluator
from src.eval.strong_reject_evaluator import StrongRejectEvaluator
from gserve.configs import ServeConfig, LLMConfig
from src.models import print_supported_models, load_model

from src.sample_attacks import SoftPrompt
from src.univ_attacks import IML
from src.adv_model import AdvModel
from src.initialize import Initializer
from src.config import GenConfig, StopCriteria
from src.eval import Evaluator
from src.activ_extractor import ActivationExtractor


logger = create_logger(__name__)


def prepare_environment(seed: int | None = 42):
    torch.set_float32_matmul_precision("high")
    env.prepare_environment()

    if seed is None:
        seed = random.randint(0, 10000)

    env.set_seed(seed)
    logger.info(f"Random seed: {seed}")


def load_data(train_ratio: float = 0.65) -> tuple[DF_Batcher, DF_Batcher]:
    data = pd.read_csv("/home/fre.gilad/source/llm-iml/data/HarmBench/harmful_behaviors.csv")
    data = data.rename(columns={"goal": "prompt"})

    # shuffle the data
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)

    split = int(train_ratio * len(data))
    ds_train = data.iloc[:split].copy()
    ds_eval = data.iloc[split:].copy()

    dl_train = DF_Batcher(ds_train, batch_size=10, shuffle=True)
    dl_eval = DF_Batcher(ds_eval, batch_size=25, shuffle=False)

    logger.info(f"Train dataset size: {len(ds_train)}")
    logger.info(f"Eval dataset size: {len(ds_eval)}")

    return dl_train, dl_eval


def load_evaluators() -> list[Evaluator]:
    evaluators: list[Evaluator] = [
        # HarmBenchEvaluator(
        #     serve_config=ServeConfig(
        #         gpu_ids=[1],
        #         startup_timeout=10 * 60,
        #         client_timeout=60,
        #         verbose=False,
        #     ),
        # ),
        StrongRejectEvaluator(
            serve_config=ServeConfig(gpu_ids=[1], startup_timeout=20 * 60, client_timeout=60),
        ),
        TemplateEvaluator(),
    ]

    logger.info(f"Evaluators loaded: {[ev.name for ev in evaluators]}")
    return evaluators


def create_model(model_name: str) -> AdvModel:
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name)
    logger.info(f"Model architecture: {model}")

    adv_model = AdvModel(model=model, tokenizer=tokenizer, num_tokens=20)
    # Initializer.normal(adv_model, std=0.1)
    Initializer.from_string(adv_model, "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !", strict=False)
    # Initializer.from_mean_std(adv_model)
    return adv_model


def run_attack(adv_model: AdvModel, evaluators: list[Evaluator], dl_train: DF_Batcher, dl_eval: DF_Batcher):
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

    univ_attack = IML(
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

    stop = StopCriteria(
        max_epochs=200,
        max_time=60 * 60 * 10,
    )

    try:
        logger.info("Starting attack...")
        adv_model = univ_attack.fit(dl_train=dl_train, dl_eval=dl_eval, stop_criteria=stop)

        logger.info("Final Eval...")
        metrics = univ_attack.evaluate(adv_model, evaluators, dl_eval, gen_config=gen_config)
        univ_attack.metric_logger.log_metrics(metrics)
        univ_attack.metric_logger.cm_task.upload_artifact(name="eval_result", artifact_object=dl_eval.df)

    except KeyboardInterrupt:
        logger.info("Interrupted by user, stopping...")

    finally:
        univ_attack.close()
        for eval in evaluators:
            eval.close()


def main():
    prepare_environment()
    dl_train, dl_eval = load_data()
    evaluators = load_evaluators()
    adv_model = create_model("meta-llama/Llama-2-7b-chat-hf")
    run_attack(adv_model, evaluators, dl_train, dl_eval)


if __name__ == "__main__":
    main()
