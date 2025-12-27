"""Validations."""

from nebula.auxiliaries import ensure_flat_list
from nebula.base import Transformer
from nebula.nw_util import df_is_empty

__all__ = [
    "AssertContainsColumns",
    "AssertCount",
    "AssertNotEmpty",
]


class AssertContainsColumns(Transformer):
    def __init__(self, *, columns: str | list[str]):
        """Raise AssertionError if the dataframe does not contain the expected columns.

        Args:
            columns (str | list(str)):
                The column or list of columns to check for existence.

        Raises:
            AssertionError if the dataframe does not contain the expected columns.
        """
        super().__init__()
        self._cols: list[str] = ensure_flat_list(columns)

    def _transform_nw(self, df):
        actual_cols = set(df.columns)
        missing_cols = [i for i in self._cols if i not in actual_cols]

        if missing_cols:
            msg = f"Missing required columns: {missing_cols}. "
            msg += f"Available columns: {sorted(actual_cols)}"
            raise AssertionError(msg)

        return df


class AssertCount(Transformer):
    def __init__(
        self,
        *,
        expected: int | None = None,
        min_count: int | None = None,
        max_count: int | None = None,
    ):
        """Assert DataFrame has expected number of rows.

        Args:
            expected: Exact number of rows expected (mutually exclusive with min/max)
            min_count: Minimum number of rows required
            max_count: Maximum number of rows allowed

        Raises:
            AssertionError: If expected is provided with min_count or max_count
            AssertionError: If row count doesn't meet expectations
            ValueError: If no count condition is specified

        Example:
            AssertCount(expected=1000)
            AssertCount(min_count=1, max_count=10000)
        """
        if expected is not None:
            if min_count is not None or max_count is not None:
                raise AssertionError(
                    "'expected' cannot be used with 'min_count' or 'max_count'"
                )

        if min_count is not None and max_count is not None:
            if min_count > max_count:
                raise AssertionError("'min_count' must be <= 'max_count'")

        super().__init__()
        self._expected: int | None = expected
        self._min: int | None = min_count
        self._max: int | None = max_count

    def _validate_count(self, count: int):
        """Validate count against expectations."""
        errors = []

        if self._expected is not None:
            if count != self._expected:
                errors.append(f"Expected exactly {self._expected} rows, got {count}")

        if self._min is not None:
            if count < self._min:
                errors.append(f"Expected at least {self._min} rows, got {count}")

        if self._max is not None:
            if count > self._max:
                errors.append(f"Expected at most {self._max} rows, got {count}")

        if errors:
            raise AssertionError(". ".join(errors))

    def _transform_pandas(self, df):
        self._validate_count(df.shape[0])
        return df

    def _transform_polars(self, df):
        import polars as pl

        if isinstance(df, pl.LazyFrame):
            count = len(df.collect())
        else:
            count = len(df)
        self._validate_count(count)
        return df

    def _transform_spark(self, df):
        self._validate_count(df.count())
        return df


class AssertNotEmpty(Transformer):
    def __init__(self, *, df_name: str = "DataFrame"):
        """Raise AssertionError if the dataframe is empty.

        Args:
            df_name: Name to use in the error message (default: "DataFrame")

        Raises:
            AssertionError: If dataframe has no rows
        """
        super().__init__()
        self._df_name: str = df_name

    def transform(self, df):
        if df_is_empty(df):  # Directly handle the dataframe type
            raise AssertionError(f"{self._df_name} is empty")
        return df
