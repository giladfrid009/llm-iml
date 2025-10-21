from src.utils.logging import create_logger
from typing import Any
import datasets
from datasets import DatasetDict, Dataset
from enum import Enum
import pandas as pd

logger = create_logger(__name__)


class DatasetName(str, Enum):
    HARMBENCH = "harmbench"
    HARMBENCH_STANDARD = "harmbench-std"
    HARMBENCH_CONTEXT = "harmbench-ctx"
    ADVBENCH = "advbench"
    ADVBENCH_SMALL = "advbench-small"
    JAILBREAK_BENCH = "jailbreak-bench"
    MALICIOUS_INSTRUCT = "malicious-instruct"
    JAILBREAK_DISTILL = "jailbreak-distill"
    WILDGUARD_MIX = "wildguard-mix"
    BEAVERTAILS = "beaver-tails"


SUPPORTED_DATASETS = [e.value for e in DatasetName]


def load_raw_dataset(name: str) -> DatasetDict:
    """
    Loads and returns a dataset by name. Each dataset is expected to have the following columns:
    - "prompt": The input prompt to the model.
    - "target": The expected output or label for the prompt.

    If the dataset does not have predefined splits, it will be loaded as a single "train" split.
    The user can then split it into training, validation, and test sets as needed.

    Args:
        name (str): The name of the dataset to load. Must be one of the supported datasets.

    Returns:
        DatasetDict: A dictionary-like object containing the dataset splits.
    """

    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Supported datasets are: {SUPPORTED_DATASETS}")

    if name == DatasetName.HARMBENCH:
        # source: https://github.com/centerforaisafety/HarmBench/tree/main/data/behavior_datasets
        return datasets.load_dataset(f"data/{DatasetName.HARMBENCH.value}")  # type: ignore

    if name == DatasetName.HARMBENCH_STANDARD:
        # source: https://github.com/centerforaisafety/HarmBench/tree/main/data/behavior_datasets
        return datasets.load_dataset(f"data/{DatasetName.HARMBENCH_STANDARD.value}")  # type: ignore

    if name == DatasetName.HARMBENCH_CONTEXT:
        # NOTE: no splits, only train
        # source: https://github.com/centerforaisafety/HarmBench/tree/main/data/behavior_datasets
        return datasets.load_dataset(f"data/{DatasetName.HARMBENCH_CONTEXT.value}")  # type: ignore

    if name == DatasetName.ADVBENCH:
        # source: https://huggingface.co/datasets/walledai/AdvBench
        return datasets.load_dataset(f"data/{DatasetName.ADVBENCH.value}")  # type: ignore

    if name == DatasetName.ADVBENCH_SMALL:
        # IRIS paper uses this small subset for training
        # NOTE: no splits, only train
        # source: https://github.com/patrickrchao/JailbreakingLLMs/blob/main/data/harmful_behaviors_custom.csv
        return datasets.load_dataset(f"data/{DatasetName.ADVBENCH_SMALL.value}")  # type: ignore

    if name == DatasetName.JAILBREAK_BENCH:
        # NOTE: no splits, only train
        # NOTE: some overlap with AdvBench and HarmBench
        # source: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
        return datasets.load_dataset(f"data/{DatasetName.JAILBREAK_BENCH.value}")  # type: ignore

    if name == DatasetName.MALICIOUS_INSTRUCT:
        # NOTE: no splits, only train
        # source: https://github.com/sj21j/Regularized_Relaxation/blob/master/data/MaliciousInstruct/harmful_behaviors.csv
        # without labels: https://huggingface.co/datasets/walledai/MaliciousInstruct
        return datasets.load_dataset(f"data/{DatasetName.MALICIOUS_INSTRUCT.value}")  # type: ignore

    if name == DatasetName.JAILBREAK_DISTILL:
        # NOTE: Some overlap with HarmBench
        # source: https://huggingface.co/datasets/jackzhang/JBDistill-Bench
        return datasets.load_dataset(f"data/{DatasetName.JAILBREAK_DISTILL.value}")  # type: ignore

    if name == DatasetName.WILDGUARD_MIX:

        def filter_fn(d: dict[str, Any]) -> bool:
            return (
                d["prompt_harm_label"] == "harmful"
                and d["response_harm_label"] == "harmful"
                and isinstance(d["response"], str)
                and not d["response"].strip().startswith("[")
            )

        ds_train: Dataset = datasets.load_dataset("allenai/wildguardmix", name="wildguardtrain", split="train")  # type: ignore
        ds_train = ds_train.filter(filter_fn).map(lambda x: {"target": x["response"][:10], **x})

        # split eval from train, eval of size 100
        split_dict = ds_train.train_test_split(test_size=100, seed=42)
        ds_train = split_dict["train"]
        ds_eval = split_dict["test"]

        ds_test: Dataset = datasets.load_dataset("allenai/wildguardmix", name="wildguardtest", split="test")  # type: ignore
        ds_test = ds_test.filter(filter_fn).map(lambda x: {"target": x["response"][:10], **x})

        return DatasetDict({"train": ds_train, "validation": ds_eval, "test": ds_test})

    if name == DatasetName.BEAVERTAILS:

        def filter_fn(d: dict[str, Any]) -> bool:
            forbidden_categories = [
                "animal_abuse",
                "child_abuse",
                "drug_abuse,weapons,banned_substance",
                "financial_crime,property_crime,theft",
                "terrorism,organized_crime",
                "violence,aiding_and_abetting,incitement",
                "self_harm",
                # "controversial_topics,politics",
                # "discrimination,stereotype,injustice",
                # "hate_speech,offensive_language",
                # "misinformation_regarding_ethics,laws_and_safety",
                # "non_violent_unethical_behavior",
                # "privacy_violation",
                # "sexually_explicit,adult_content",
            ]

            return d["is_safe"] is False and any(d[cat] is True for cat in forbidden_categories)

        ds_train: Dataset = datasets.load_dataset("PKU-Alignment/BeaverTails", split="30k_train")  # type: ignore
        ds_train = ds_train.filter(filter_fn).rename_column("response", "target")
        ds_train = ds_train.map(lambda x: {"target": x["target"][:10], **x})

        # split eval from train, eval of size 100
        split_dict = ds_train.train_test_split(test_size=100, seed=42)
        ds_train = split_dict["train"]
        ds_eval = split_dict["test"]

        ds_test: Dataset = datasets.load_dataset("PKU-Alignment/BeaverTails", split="30k_test")  # type: ignore
        ds_test = ds_test.filter(filter_fn).rename_column("response", "target")
        ds_test = ds_test.map(lambda x: {"target": x["target"][:10], **x})

        return DatasetDict({"train": ds_train, "validation": ds_eval, "test": ds_test})

    raise ValueError(f"Unsupported dataset: {name}")


def load_dataset(name: str, shuffle: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads a dataset by name and returns the training, validation, and test sets as pandas DataFrames.

    Args:
        names (list[str]): List of dataset names to load. Each name must be one of the supported datasets.
        shuffle (bool): Whether to shuffle the dataset splits. Default is True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and test datasets as pandas DataFrames.
    """

    ds_dict = load_raw_dataset(name)

    ds_train = ds_dict["train"]
    ds_val = ds_dict["validation"] if "validation" in ds_dict else None
    ds_test = ds_dict["test"] if "test" in ds_dict else None

    if ds_val is None and ds_test is None:
        ds_val = ds_train.shuffle(seed=0, keep_in_memory=True)
        ds_test = ds_train.shuffle(seed=1, keep_in_memory=True)
        logger.info(f"Dataset {name} has no validation or test set, using training set for both.")

    if ds_val is None and ds_test is not None:
        ds_val = ds_test.shuffle(seed=2, keep_in_memory=True)
        logger.info(f"Dataset {name} has no validation set, using test set as validation set.")

    if ds_test is None and ds_val is not None:
        ds_test = ds_val.shuffle(seed=3, keep_in_memory=True)
        logger.info(f"Dataset {name} has no test set, using validation set as test set.")

    if shuffle:
        ds_train = ds_train.shuffle(keep_in_memory=True)  # type: ignore
        ds_val = ds_val.shuffle(keep_in_memory=True)  # type: ignore
        ds_test = ds_test.shuffle(keep_in_memory=True)  # type: ignore

    return (
        ds_train.to_pandas().reset_index(drop=True),  # type: ignore
        ds_val.to_pandas().reset_index(drop=True),  # type: ignore
        ds_test.to_pandas().reset_index(drop=True),  # type: ignore
    )
