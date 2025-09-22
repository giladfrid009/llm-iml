import torch
import sys
import pathlib
import pandas as pd

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.utils import env
from src.data import DF_Batcher
from src.eval.harmbench_evaluator import HarmBenchEvaluator
from src.eval.template_evaluator import TemplateEvaluator
from src.eval.strong_reject_evaluator import StrongRejectEvaluator
from gserve.configs import ServeConfig, LLMConfig
from notebooks.utils import print_supported_models, load_model

from src.univ_attacks import UnivSoftPrompt
from src.adv_model import AdvModel
from src.initialize import Initializer
from src.config import GenConfig, StopCriteria
from src.fgsm_optim import FGSM
from src.eval import Evaluator


def run_attack(adv_model: AdvModel, evaluators: list[Evaluator], dl_train: DF_Batcher, dl_eval: DF_Batcher):
    gen_config = GenConfig(
        max_new_tokens=512,
        do_sample=False,
        remove_invalid_values=True,
    )

    optimizer = FGSM(
        adv_model.parameters(),
        lr=0.001,
    )

    univ_attack = UnivSoftPrompt(
        adv_model=adv_model,
        optimizer=optimizer,
        evaluators=evaluators,
        eval_freq=10,
        gen_config=gen_config,
        mixed_precision=False,
        log_dir="logs",
    )

    stop = StopCriteria(
        max_epochs=200,
        max_time=60 * 60,
    )

    adv_model = univ_attack.fit(dl_train, dl_eval, stop_criteria=stop)

    univ_attack.close()


def load_data() -> tuple[DF_Batcher, DF_Batcher]:
    data = pd.read_csv("/home/fre.gilad/source/llm-iml/data/HarmBench/harmful_behaviors.csv")
    data = data.rename(columns={"goal": "prompt"})
    data = data.sample(frac=1).reset_index(drop=True)
    
    split = int(0.65 * len(data))
    ds_train = data.iloc[:split].copy()
    ds_eval = data.iloc[split:].copy()

    dl_train = DF_Batcher(ds_train, batch_size=10, shuffle=True)
    dl_eval = DF_Batcher(ds_eval, batch_size=25, shuffle=False)

    print("Train dataset size:", len(ds_train))
    print("Eval dataset size:", len(ds_eval))

    return dl_train, dl_eval


def main():
    torch.set_float32_matmul_precision("high")

    env.prepare_environment()
    env.set_seed(42)
    
    dl_train, dl_eval = load_data()

    evaluators = [
        HarmBenchEvaluator(
            serve_config=ServeConfig(
                gpu_ids=[1],
                startup_timeout=10 * 60,
                client_timeout=60,
                verbose=True,
            ),
        ),
        # StrongRejectEvaluator(
        #     serve_config=ServeConfig(gpu_ids=[2], startup_timeout=20 * 60, client_timeout=60),
        #     binary_thresh=0.5,
        # ),
        TemplateEvaluator(),
    ]

    print_supported_models()

    model, tokenizer = load_model("meta-llama/Llama-2-7b-chat-hf")

    print(model)

    adv_model = AdvModel(model=model, tokenizer=tokenizer, num_tokens=20)

    Initializer.from_string(adv_model, "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !", strict=False)

    run_attack(adv_model, evaluators, dl_train, dl_eval)


if __name__ == "__main__":
    main()
