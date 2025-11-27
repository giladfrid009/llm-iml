# NOTE: GPT5 Generated

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.utils.logging import create_logger, setup_logging, loglevel_names  # noqa: E402


logger = create_logger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _display_width() -> int:
    """Best-effort detection of terminal width, with a safe fallback."""
    try:
        import shutil

        return shutil.get_terminal_size().columns
    except Exception:
        return 120


def _read_df(path: pathlib.Path, **kwargs) -> pd.DataFrame | None:
    """Read a dataframe from various supported formats based on file suffix."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, lines=False, **kwargs)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".feather":
        return pd.read_feather(path, **kwargs)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path, **kwargs)

    return None


def _finite_only(s: pd.Series) -> pd.Series:
    """Return only finite (non-NaN, non-inf) values from a series."""
    s = pd.to_numeric(s, errors="coerce")
    return s[np.isfinite(s)]


def _safe_name_from_path(path_str: str) -> str:
    """
    Turn a (relative) path like 'subdir/foo-bar.csv' into a filesystem-safe
    name such as 'subdir__foo-bar.csv' (suitable as part of a filename).

    This also helps avoid collisions when using --recurse and having files
    with the same basename in different folders.
    """
    normalized = path_str.replace("\\", "/").strip("/")
    normalized = normalized.replace("/", "__")
    normalized = normalized.replace(" ", "_")
    return normalized.strip()


def _select_numeric_columns(df: pd.DataFrame, metrics: list[str] | None) -> pd.Index:
    """
    Select numeric columns from the dataframe.

    - If `metrics` is None, return all numeric columns.
    - If `metrics` is provided, return only those that exist in the dataframe
      and are numeric.
    """
    if metrics is None:
        return df.select_dtypes(include=["number"]).columns

    existing = [m for m in metrics if m in df.columns]
    if not existing:
        return pd.Index([])

    return df[existing].select_dtypes(include=["number"]).columns


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "data_path",
        type=str,
        nargs="+",
        metavar="PATH",
        help=(
            "Paths to evaluation data file or directory containing data files. "
            "Supported formats: CSV, JSON, JSONL, Excel, TSV, Parquet, Feather, Pickle. "
            "If a directory is provided, all files in the directory will be processed."
        ),
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=loglevel_names(),
        default="INFO",
        metavar="LEVEL",
        help=f"Logging level to python-logger. Available levels: {loglevel_names()}",
    )
    parser.add_argument(
        "--recurse",
        action="store_true",
        help="Recursively search directories for data files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="summaries",
        metavar="DIR",
        help="Directory to save numeric summary tables.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        metavar="COL",
        help=("Optional list of numeric column names (metrics) to summarize. If omitted, all numeric columns are summarized."),
    )

    args = parser.parse_args()

    print("\nParsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    return args


def read_data(paths: list[str], recurse: bool) -> tuple[list[str], list[pd.DataFrame]]:
    path_list = []
    data_list = []

    user_dir = pathlib.Path.cwd()
    for path in paths:
        # make all paths relative to CWD
        path = pathlib.Path(path).resolve()
        path = path.relative_to(user_dir, walk_up=True)

        if path.is_dir():
            sub_paths, sub_data = read_data([p.as_posix() for p in path.iterdir() if p.is_file() or recurse], recurse)
            if len(sub_data) == 0:
                continue

            logger.info(f"Loaded {len(sub_data)} files from directory: {path}")
            path_list.extend(sub_paths)
            data_list.extend(sub_data)
            continue

        try:
            df = _read_df(path)
            if df is None:
                continue

        except Exception as e:
            logger.error(f"Error reading file {path}: {e}. Skipping...")
            continue

        req_cols = {"prompt", "response"}
        if not req_cols.issubset(df.columns):
            logger.error(f"File {path} is missing required columns: {req_cols}. Skipping...")

        path_list.append(path.as_posix())
        data_list.append(df)

    return path_list, data_list


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize_per_dataset(
    path_list: list[str],
    data_list: list[pd.DataFrame],
    width: int,
    output_dir: pathlib.Path,
    metrics: list[str] | None,
) -> None:
    """
    Print and save a numeric summary for each dataset separately.

    For each dataset:
      - compute mean, std, median, count over finite values only
      - print a table to stdout
      - save a human-readable JSON file in output_dir

    If `metrics` is not None, only those columns are considered (when present and numeric).
    """
    for ds_name, df in zip(path_list, data_list):
        numeric_cols = _select_numeric_columns(df, metrics)

        if len(numeric_cols) == 0:
            if metrics is None:
                logger.warning(f"Dataset {ds_name} has no numeric columns.")
            else:
                logger.warning(f"Dataset {ds_name} has no numeric columns among requested metrics: {metrics}")
            continue

        stats: dict[str, dict[str, float | int]] = {}
        for col in numeric_cols:
            s = _finite_only(df[col])
            if s.empty:
                continue
            stats[col] = {
                "mean": s.mean(),
                "std": s.std(),
                "median": s.median(),
                "count": int(s.size),
            }

        if not stats:
            logger.warning(f"Dataset {ds_name} has no finite numeric values for the selected columns.")
            continue

        summary = pd.DataFrame.from_dict(stats, orient="index").loc[:, ["mean", "std", "median", "count"]].sort_index().round(2)

        print()
        print("=".center(width, "="))
        print(f"Numeric Summary for Dataset: {ds_name}".center(width))
        if metrics is not None:
            print(f"(metrics: {', '.join(metrics)})".center(width))
        print("=".center(width, "="))

        with pd.option_context("display.width", width, "display.max_rows", 1000, "display.max_columns", 1000):
            print(summary)

        safe_name = _safe_name_from_path(ds_name)

        json_obj = {
            "dataset": ds_name,
            "metrics_filter": metrics,
            "columns": [
                {
                    "column": idx,
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "median": float(row["median"]),
                    "count": int(row["count"]),
                }
                for idx, row in summary.iterrows()
            ],
        }

        json_path = output_dir / f"{safe_name}_summary.json"
        try:
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save per-dataset summary JSON for {ds_name}: {e}")


def summarize_overall(
    path_list: list[str],
    data_list: list[pd.DataFrame],
    width: int,
    output_dir: pathlib.Path,
    metrics: list[str] | None,
) -> None:
    """
    Build a simple mean-only summary across all datasets and print/save it.

    Final table:
      - rows: datasets (unique, based on path)
      - columns: numeric metrics
      - cells: mean over finite values only (NaN if metric not present)

    Saved artifact (unchanged by display compaction):
      - numeric_summary_overall.json
        {
          "metrics_filter": [...],
          "datasets": {
            "<dataset_id>": {
              "metric_a": 0.1234,
              "metric_b": null,
              ...
            },
            ...
          }
        }
    """
    records: list[dict[str, object]] = []

    for ds_name, df in zip(path_list, data_list):
        numeric_cols = _select_numeric_columns(df, metrics)
        if len(numeric_cols) == 0:
            continue

        for col in numeric_cols:
            s = _finite_only(df[col])
            if s.empty:
                continue
            records.append(
                {
                    "dataset": ds_name,
                    "metric": col,
                    "mean": s.mean(),
                }
            )

    print()
    print("=".center(width, "="))
    print("Overall Mean Values Across Datasets".center(width))
    if metrics is not None:
        print(f"(metrics: {', '.join(metrics)})".center(width))
    print("=".center(width, "="))

    if not records:
        if metrics is None:
            print("No numeric columns with finite values found across any dataset.")
        else:
            print(f"No finite numeric values found for the requested metrics across any dataset: {metrics}")
        return

    overall_df = pd.DataFrame.from_records(records)

    # Long -> wide: rows = dataset, columns = metric, values = mean
    wide_df = (
        overall_df.pivot(index="dataset", columns="metric", values="mean")
        .sort_index()  # sort datasets
        .reindex(sorted(overall_df["metric"].unique()), axis=1)  # sort metrics
        .round(2)
        .reset_index()  # make 'dataset' a normal column
    )

    # --------- Pretty printing with optional compaction ---------
    with pd.option_context("display.width", width, "display.max_rows", 1000, "display.max_columns", 1000):
        display_df = wide_df.copy()

        # First attempt: full table
        text = display_df.to_string(index=False, na_rep=" -- ")
        lines = text.splitlines()
        too_wide = any(len(line) > width for line in lines)

        legend: dict[str, str] = {}
        truncated_datasets = False

        if too_wide:
            # 1) Truncate dataset names from the left, keep the tail
            max_ds_len = max(20, min(40, width // 3))  # heuristic

            def _truncate_ds(s: str) -> str:
                s = str(s)
                if len(s) <= max_ds_len:
                    return s
                return "…" + s[-(max_ds_len - 1) :]

            display_df["dataset"] = display_df["dataset"].apply(_truncate_ds)
            truncated_datasets = any(d != o for d, o in zip(display_df["dataset"], wide_df["dataset"], strict=False))

            # 2) Replace metric column names with short aliases (m1, m2, ...)
            metric_cols = [c for c in display_df.columns if c != "dataset"]
            legend = {f"M{i}": col for i, col in enumerate(metric_cols, start=1)}
            rename_map = {col: alias for alias, col in legend.items()}
            display_df = display_df.rename(columns=rename_map)

            # Re-render after compaction
            text = display_df.to_string(index=False, na_rep=" -- ")

        # If we compacted, print legend first
        if legend:
            print("Compact view legend (columns):")
            for alias, original in legend.items():
                print(f"  {alias}: {original}")
            if truncated_datasets:
                print("Note: 'dataset' values may be truncated on the left (…suffix).")
            print()

        print(text)

    # --------- JSON export (uses original wide_df, not compacted view) ---------
    hierarchical: dict[str, dict[str, float | None]] = {}
    for _, row in wide_df.iterrows():
        ds: str = row["dataset"]
        hierarchical[ds] = {}
        for col, val in row.items():
            if col == "dataset":
                continue
            else:
                hierarchical[ds][col] = float(val)

    json_obj = {
        "metrics_filter": metrics,
        "datasets": hierarchical,
    }

    json_path = output_dir / "numeric_summary_overall.json"
    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2)
        logger.info(f"Saved overall numeric summary JSON to: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save overall numeric summary JSON: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    width = _display_width()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path_list, data_list = read_data(args.data_path, args.recurse)
    if not data_list:
        logger.error("No valid datasets loaded. Exiting.")
        return

    metrics = args.metrics

    summarize_per_dataset(
        path_list=path_list,
        data_list=data_list,
        width=width,
        output_dir=output_dir,
        metrics=metrics,
    )

    summarize_overall(
        path_list=path_list,
        data_list=data_list,
        width=width,
        output_dir=output_dir,
        metrics=metrics,
    )


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level)
    main(args)
