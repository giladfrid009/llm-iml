import subprocess
import argparse
import sys
import copy
from typing import Dict, List, Any
import torch
import gc
# =============================================================================
# Global Constants & Configuration
# =============================================================================

PROJECT_NAME = "UAP"
SCRIPT_PATH = "scripts/run_uap.py"

# Default Arguments shared across all runs unless overridden
DEFAULT_RUN_ARGS = {
    "log_dir": "logs-uap-judge-strongreject",
    "dataset": "advbench",
    "test_datasets": ["advbench", "harmbench-std"],
    "evaluator": ["strong-reject", "keyword-matching"],
    "eval_freq": 0.2,
    "eval_batch": 10,
}

# =============================================================================
# Experiment Setups
# =============================================================================

# Define reusable experiment configurations
# Using clear names for standard setups.
# These dictionaries are essentially just sets of CLI arguments.

EXP_SETUP_PLACEHOLDER = dict(
    illegal_arg="value",  # Placeholder for structure
)

EXP_SETUP_2 = dict(
    skip_fooled="true",
    skip_failed="true",
    target_controls="false",
    # inner-attack params
    inner_attack_lr=5e-3,
    inner_attack_steps=50,
    inner_attack_target_matching="false",
)

EXP_SETUP_3 = dict(
    skip_fooled="true",
    skip_failed="true",
    target_controls="false",
    # inner-attack params
    inner_attack_lr=5e-3,
    inner_attack_steps=100,
    inner_attack_target_matching="true",
)

EXP_SETUP_5 = dict(
    skip_fooled="true",
    skip_failed="true",
    target_controls="false",
    # inner-attack params
    inner_attack_lr=5e-3,
    inner_attack_steps=150,
    inner_attack_target_matching="false",
)

EXP_SETUP_6 = dict(
    skip_fooled="true",
    skip_failed="true",
    target_controls="false",
    # inner attack params
    inner_attack_steps=75,
    inner_attack_target_matching="false",
)

EXP_SETUP_8 = dict(
    skip_fooled="true",
    skip_failed="true",
    target_controls="false",
    # inner-attack params
    inner_attack_lr=5e-3,
    inner_attack_steps=7,
    inner_attack_target_matching="false",
)

EXP_SETUP_12 = dict(
    skip_fooled="true",
    skip_failed="true",
    target_controls="false",
    # inner-attack params
    inner_attack_lr=5e-3,
    inner_attack_steps=35,
    inner_attack_target_matching="false",
)


# =============================================================================
# Model Registry
# =============================================================================
# Structure:
#   key: Internal identifier
#   value: Dict containing:
#       - "nick": (Required) Used for run_name generation
#       - "experiments": (Required) Dict of {experiment_suffix: specific_args_dict}
#       - ...Any other key represents a CLI argument for the script

MODELS = {
    "llama_2": {
        "nick": "uap-llama2",
        "experiments": {"setup8": EXP_SETUP_8},
        # CLI Arguments
        "model": "meta-llama/Llama-2-7b-chat-hf",
    },
    "llama_32": {
        "nick": "uap-llama-32",
        "experiments": {"setup1": EXP_SETUP_2},
        # CLI Arguments
        "model": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "phi_4": {
        "nick": "uap-phi-4",
        "experiments": {"setup2": EXP_SETUP_2},
        # CLI Arguments
        "model": "microsoft/Phi-4-mini-instruct",
    },
    "gemma_2": {
        "nick": "uap-gemma-2",
        # "experiments": {"setup5": EXP_SETUP_5},
        "experiments": {"setup5": EXP_SETUP_12}, # NOTE: FOR HARMBENCH
        # CLI Arguments
        "model": "google/gemma-2-2b-it",
        "kv_caching": "false",
    },
    "qwen_3": {
        "nick": "uap-qwen-3",
        # "experiments": {"setup8": EXP_SETUP_8},
        "experiments": {"setup5": EXP_SETUP_12}, # NOTE: FOR HARMBENCH
        # CLI Arguments
        "model": "Qwen/Qwen3-4B-Instruct-2507",
    },
    "llama_2_cat": {
        "nick": "uap-llama2-cat",
        "experiments": {"setup8": EXP_SETUP_8},
        # CLI Arguments
        "model": "ContinuousAT/Llama-2-7B-CAT",
    },
    "mistral_r2d2": {
        "nick": "uap-mistral-r2d2",
        "experiments": {"setup3": EXP_SETUP_3},
        # CLI Arguments
        "model": "cais/zephyr_7b_r2d2",
    },
    "llama_3_rr": {
        "nick": "uap-llama3-rr",
        "experiments": {"setup12": EXP_SETUP_12},
        # CLI Arguments
        "model": "GraySwanAI/Llama-3-8B-Instruct-RR",
    },
    "llama_3_lat": {
        "nick": "uap-llama3-lat",
        "experiments": {"setup12": EXP_SETUP_12},
        # CLI Arguments
        "model": "LLM-LAT/robust-llama3-8b-instruct",
    },
    "phi_3_capo": {
        "nick": "uap-phi-3-capo",
        "experiments": {"setup12": EXP_SETUP_6},
        # CLI Arguments
        "model": "ContinuousAT/Phi-CAPO",
    },
}

# =============================================================================
# Helper Functions
# =============================================================================


def build_command(run_config: Dict[str, Any], script_path: str, run_name: str, project_name: str) -> List[str]:
    """Constructs the subprocess command list from configuration dictionary."""
    cmd_args = [
        "python",
        script_path,
        "--run_name",
        run_name,
        "--project_name",
        project_name,
    ]

    for arg_name, arg_value in run_config.items():
        cmd_args.append(f"--{arg_name}")
        # Handle lists (repeated arguments or specific formatting based on implementation)
        # Assuming argparse nargs='+' style inputs
        if isinstance(arg_value, list):
            cmd_args.extend(map(str, arg_value))
        else:
            cmd_args.append(str(arg_value))

    return cmd_args


def collect_garbage():
    """Utility to clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def run_experiment(model_key: str, iteration: int, total_iters: int, name_suffix: str):
    """Executes experiments for a single model configuration."""

    # 1. Retrieve Model Configuration
    # Safe copy to avoid side effects
    model_conf = copy.deepcopy(MODELS[model_key])

    # Extract special control fields
    model_nick = model_conf.pop("nick", model_key)
    model_experiments = model_conf.pop("experiments", {})

    print(f"\n>> Processing Model: {model_key} ({model_nick})")

    if not model_experiments:
        print(f"!! No experiments defined for {model_key}. Skipping.")
        return

    # 2. Iterate over experiments defined for this model
    for exp_setup_name, exp_params in model_experiments.items():
        full_run_name = f"{model_nick}-{exp_setup_name}{name_suffix}-iter{iteration}"

        print("=" * 60)
        print(f"▶️ Running Experiment: {full_run_name} ({iteration}/{total_iters})")
        print("=" * 60)

        # 3. Construct Run Parameters Strategy:
        #    Global Defaults
        #    -> MERGED WITH -> Model Params
        #    -> MERGED WITH -> Experiment Params

        run_params = copy.deepcopy(DEFAULT_RUN_ARGS)

        # Add all remaining model config items as CLI arguments
        run_params.update(model_conf)

        # Add experiment-specific overrides
        run_params.update(exp_params)

        # 4. Build and Execute Command
        cmd = build_command(run_config=run_params, script_path=SCRIPT_PATH, run_name=full_run_name, project_name=PROJECT_NAME)

        try:
            # print("Executing:", " ".join(cmd))
            collect_garbage()
            subprocess.run(cmd, shell=False, check=True, start_new_session=True)
        except KeyboardInterrupt:
            print("\n🚨 Execution Interrupted by User")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during execution of {full_run_name}: {e}")
            # Continue to next experiment/model
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run consolidated UAP experiments.")
    parser.add_argument("--models", nargs="+", default=["all"], choices=list(MODELS.keys()) + ["all"], help="List of model keys to run, or 'all'")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix to append to experiment names.")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations to run each experiment.")

    args = parser.parse_args()

    target_models = list(MODELS.keys()) if "all" in args.models else args.models

    print(f"🚀 Starting UAP Runs for models: {target_models}")
    print(f"📋 Global Default Args: {DEFAULT_RUN_ARGS}")

    for model_key in target_models:
        for iteration in range(1, args.iters + 1):
            print(f"\n--- Iteration {iteration} / {args.iters} ---")
            run_experiment(model_key, iteration, args.iters, args.name_suffix)
