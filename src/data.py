import random
from typing import Any, Generator
import pandas as pd


class TableLoader:
    """
    Iterate over a DataFrame in row-wise batches.

    Standard columns which are expected to be present in the DataFrame:
    - 'prompt' (str): Input prompt for the model.
    - 'response' (str): Model response or output, should be present for evaluation.
    - 'target' (str): Target response of the model, should be present for training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Precompute index list and number of samples
        self._indices = list(df.index)
        self._n_samples = len(self._indices)
        self._n_batches = self._compute_num_batches()

    def get_hparams(self) -> dict:
        """
        Returns the hyperparameters of the TableLoader as a dictionary.
        """
        return {
            "num_samples": self._n_samples,
            "num_batches": self._n_batches,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }

    def validate(self, columns: list[str]) -> None:
        """
        Public method: check that each column in `columns` exists in `self.df.columns`.

        Raises:
            ValueError: if any requested column is not found in self.df.
        """
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

    @property
    def n_samples(self) -> int:
        """
        Number of samples in the DataFrame.
        """
        return self._n_samples

    def _compute_num_batches(self) -> int:
        if self.drop_last:
            return self._n_samples // self.batch_size
        else:
            # include a final smaller batch if necessary
            return (self._n_samples + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        """
        Number of batches (independent of which columns you slice out).
        """
        return self._n_batches

    def __iter__(self) -> Generator[dict[str, list[Any]], None, None]:
        columns = self.df.columns.tolist()

        # Make a copy of the indices and shuffle if requested
        idxs = self._indices.copy()
        if self.shuffle:
            random.shuffle(idxs)

        # Now slice out row-indices batch by batch
        for batch_idx in range(self._n_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_idxs = idxs[start:end]

            # If drop_last=True and the final batch is too small, stop early
            if len(batch_idxs) < self.batch_size and self.drop_last:
                break

            data = {col: self.df.loc[batch_idxs, col].tolist() for col in columns}
            yield data

    def copy(self, **kwargs) -> "TableLoader":
        """
        Create and return a shallow-copy of this TableLoader. The new instance
        wraps the same DataFrame (no deep copy of df), but has its own attributes
        (columns, batch_size, shuffle, drop_last).

        Args:
            **kwargs: Additional keyword arguments to override attributes in the new instance.
        """
        attrs = {
            "df": self.df,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }

        for key, value in kwargs.items():
            if key in attrs:
                attrs[key] = value
            else:
                raise ValueError(f"Invalid attribute '{key}' for TableLoader")

        return TableLoader(**attrs)

    def set_column(self, col_name: str, values: list[Any]) -> None:
        """
        Add or override a column in the DataFrame with the specified name and values.

        Args:
            col_name (str): Name of the new column.
            values (list[Any]): Values to be added in the new column.
        """

        if self.shuffle or self.drop_last:
            raise ValueError("Cannot add column while `shuffle` or `drop_last` is enabled")

        if len(values) != len(self.df):
            raise ValueError(f"Length of values ({len(values)}) does not match DataFrame length ({len(self.df)})")

        self.df[col_name] = values
