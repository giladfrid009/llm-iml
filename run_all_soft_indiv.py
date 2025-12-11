import subprocess
import argparse


class RunConfig:
    def __init__(self, script_path: str, config: dict):
        self.script_path = script_path
        self.config = config


dataset_name = "advbench"
evaluator_name = ["strong-reject"]
# evaluator_name = ["hb-judge"]

CONFIG_DICT = {
    # non-robust models
    # "indiv-soft-llama2-7b": RunConfig(
    #     script_path="run_soft_indiv.py",
    #     config={
    #         "model": "meta-llama/Llama-2-7b-chat-hf",
    #         "dataset": dataset_name,
    #         "evaluator": evaluator_name,
    #     },
    # ),
    # "indiv-soft-llama3.2-3b": RunConfig(
    #     script_path="run_soft_indiv.py",
    #     config={
    #         "model": "meta-llama/Llama-3.2-3B-Instruct",
    #         "dataset": dataset_name,
    #         "evaluator": evaluator_name,
    #     },
    # ),
    # "indiv-soft-phi4-mini": RunConfig(
    #     script_path="run_soft_indiv.py",
    #     config={
    #         "model": "microsoft/Phi-4-mini-instruct",
    #         "dataset": dataset_name,
    #         "evaluator": evaluator_name,
    #     },
    # ),
    # "indiv-soft-gemma2-2b": RunConfig(
    #     script_path="run_soft_indiv.py",
    #     config={
    #         "model": "google/gemma-2-2b-it",
    #         "dataset": dataset_name,
    #         "evaluator": evaluator_name,
    #     },
    # ),
    # "indiv-soft-qwen3-4b": RunConfig(
    #     script_path="run_soft_indiv.py",
    #     config={
    #         "model": "Qwen/Qwen3-4B-Instruct-2507",
    #         "dataset": dataset_name,
    #         "evaluator": evaluator_name,
    #     },
    # ),
    # # robust models
    # "indiv-soft-llama2-7b-cat": RunConfig(
    #     script_path="run_soft_indiv.py",
    #     config={
    #         "model": "ContinuousAT/Llama-2-7B-CAT",
    #         "dataset": dataset_name,
    #         "evaluator": evaluator_name,
    #     },
    # ),
    # "indiv-soft-mistral-7b-v0.1-r2d2": RunConfig(
    #     script_path="run_soft_indiv.py",
    #     config={
    #         "model": "cais/zephyr_7b_r2d2",
    #         "dataset": dataset_name,
    #         "evaluator": evaluator_name,
    #     },
    # ),
    "indiv-soft-llama3-8b-rr": RunConfig(
        script_path="run_soft_indiv.py",
        config={
            "model": "GraySwanAI/Llama-3-8B-Instruct-RR",
            "dataset": dataset_name,
            "evaluator": evaluator_name,
        },
    ),
    # "indiv-soft-llama3-8b-lat": RunConfig(
    #     script_path="run_soft_indiv.py",
    #     config={
    #         "model": "LLM-LAT/robust-llama3-8b-instruct",
    #         "dataset": dataset_name,
    #         "evaluator": evaluator_name,
    #     },
    # ),
    # "indiv-soft-phi3-mini-capo": RunConfig(
    #     script_path="run_soft_indiv.py",
    #     config={
    #         "model": "ContinuousAT/Phi-CAPO",
    #         "dataset": dataset_name,
    #         "evaluator": evaluator_name,
    #     },
    # ),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all SoftPrompt experiments.")

    parser.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help="Suffix to append to each experiment name.",
    )
    
    parser.add_argument(
        "--evaluator",
        type=str,
        default="strong-reject",
        help="Evaluator to use for the experiments.",
    )

    args = parser.parse_args()

    for name, config in CONFIG_DICT.items():
        print("=" * 50)
        print("=" * 50)
        print(f" Running experiment: {name}")
        print("=" * 50)
        print("=" * 50)

        # override evaluator if specified
        config.config["evaluator"] = [args.evaluator]

        # format run name with suffix
        suffix = args.name_suffix
        if suffix:
            suffix = "-" + suffix

        cmd_args = ["python", config.script_path, "--run_name", name + suffix]
        for arg_name, arg_value in config.config.items():
            cmd_args.append(f"--{arg_name}")
            if isinstance(arg_value, list):
                cmd_args.extend(map(str, arg_value))
            else:
                cmd_args.append(str(arg_value))

        try:
            subprocess.run(cmd_args, shell=False)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
            continue
