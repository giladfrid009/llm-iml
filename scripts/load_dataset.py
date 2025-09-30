from src.utils.logging import create_logger
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
    ADVBENCH_SMALL = "advbench-small"  # one used by IRIS for training
    JAILBREAK_BENCH = "jailbreak-bench"
    MALICIOUS_INSTRUCT = "malicious-instruct"


SUPPORTED_DATASETS = [e.value for e in DatasetName]


def load_single_dataset(name: str) -> Dataset:
    """
    Loads and returns a dataset with two main columns (and possibly other auxiliary columns):
    - "prompt": The input prompt to the model.
    - "target": The expected output or label for the prompt.
    """

    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Supported datasets are: {SUPPORTED_DATASETS}")

    if name == DatasetName.HARMBENCH:
        ds_dict: DatasetDict = datasets.load_dataset("data/harmbench")  # type: ignore
        return ds_dict["train"].filter(lambda x: x["functional_category"] in ["standard", "contextual"])

    if name == DatasetName.HARMBENCH_STANDARD:
        ds_dict: DatasetDict = datasets.load_dataset("data/harmbench")  # type: ignore
        return ds_dict["train"].filter(lambda x: x["functional_category"] == "standard")

    if name == DatasetName.HARMBENCH_CONTEXT:
        ds_dict: DatasetDict = datasets.load_dataset("data/harmbench")  # type: ignore
        return ds_dict["train"].filter(lambda x: x["functional_category"] == "contextual")

    if name == DatasetName.ADVBENCH:
        return datasets.load_dataset("walledai/AdvBench", split="train")  # type: ignore

    if name == DatasetName.ADVBENCH_SMALL:
        ds_dict: DatasetDict = datasets.load_dataset("data/advbench_small")  # type: ignore
        return ds_dict["train"]

    if name == DatasetName.JAILBREAK_BENCH:
        ds: Dataset = datasets.load_dataset("JailbreakBench/JBB-Behaviors", name="behaviors", split="harmful")  # type: ignore
        return ds.rename_columns({"Goal": "prompt", "Target": "target"})

    if name == DatasetName.MALICIOUS_INSTRUCT:
        ds_dict: DatasetDict = datasets.load_dataset("data/malicious_instruct")  # type: ignore
        return ds_dict["train"]

    raise ValueError(f"Unsupported dataset: {name}")


def split_data(
    full_data: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    split_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    size = len(full_data)
    val_size = int(size * val_ratio)
    test_size = int(size * test_ratio)
    train_size = size - val_size - test_size

    full_data = full_data.sample(frac=1, random_state=split_seed).reset_index(drop=True)
    ds_train = full_data.iloc[:train_size]
    ds_val = full_data.iloc[train_size : train_size + val_size]
    ds_test = full_data.iloc[train_size + val_size :]

    if len(ds_val) == 0 and len(ds_test) == 0:
        ds_val = ds_train.copy()
        ds_test = ds_train.copy()
        logger.info("Both validation and test sets are empty, using training set for both.")

    elif len(ds_val) == 0:
        ds_val = ds_test.copy()
        logger.info("Validation set is empty, using test set as validation set.")

    elif len(ds_test) == 0:
        ds_test = ds_val.copy()
        logger.info("Test set is empty, using validation set as test set.")

    ds_train = ds_train.sample(frac=1, random_state=0).reset_index(drop=True)
    ds_val = ds_val.sample(frac=1, random_state=1).reset_index(drop=True)
    ds_test = ds_test.sample(frac=1, random_state=2).reset_index(drop=True)

    return ds_train, ds_val, ds_test


def load_datasets(
    *names: str,
    val_ratio: float = 0.5,
    test_ratio: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds_list: list[pd.DataFrame] = [load_single_dataset(name).to_pandas(batched=False) for name in names]  # type: ignore

    if len(ds_list) == 1:
        ds_full = ds_list[0]

    else:
        ds_full = pd.concat(ds_list, axis=0, join="outer", ignore_index=True)
        ds_full.reset_index(drop=True, inplace=True)
        logger.info(f"Joined datasets: {list(names)}")

        orig_size = len(ds_full)
        ds_full.drop_duplicates(subset=["prompt"], keep="first", inplace=True, ignore_index=True)
        ds_full.dropna(subset=["prompt"], inplace=True, ignore_index=True)
        ds_full.reset_index(drop=True, inplace=True)
        logger.info(f"Dropped {orig_size - len(ds_full)} duplicate rows. Dataset size is now {len(ds_full)}.")

    return split_data(ds_full, val_ratio, test_ratio, split_seed=42)
