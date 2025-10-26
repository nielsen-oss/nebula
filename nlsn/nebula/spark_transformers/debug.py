"""Transformers for debugging."""

from typing import Iterable, List, Optional, Union

from nlsn.nebula.auxiliaries import assert_cmp, assert_is_integer, ensure_flat_list
from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_util import to_pandas_to_spark

__all__ = [
    "Limit",
    "LocalCheckpoint",
    "Show",
    "ShowDistinct",
    "ToPandasToSpark",
]


class Limit(Transformer):
    def __init__(self, *, n: int):
        """Limit the result count to the number specified.

        Args:
            n (n):
                Limits the result count to the number specified.
                If 0 is provided, it returns an empty dataframe.
                If -1 is provided, it returns the full dataframe.

        Raises:
        - ValueError: If 'n' is not an integer.
        - ValueError: If 'n' is < -1.
        """
        assert_is_integer(n, "n")
        assert_cmp(n, "ge", -1, "n")

        super().__init__()
        self._n: int = int(n)

    def _transform(self, df):
        if self._n == -1:
            return df

        return df.limit(self._n)


class LocalCheckpoint(Transformer):
    def __init__(self, *, eager: bool = True):
        """Return a locally checkpointed version of this DataFrame.

        Checkpointing can be used to truncate the logical plan of this
        DataFrame, which is especially useful in iterative algorithms
        where the plan may grow exponentially. Local checkpoints are
        stored in the executors using the caching subsystem, and
        therefore they are not reliable for recovery capabilities,
        but this is not the scope of this transformer.

        Args:
            eager (bool):
                Whether to checkpoint this DataFrame immediately.
                Defaults to True.
        """
        super().__init__()
        self._eager: bool = eager

    def _transform(self, df):
        return df.localCheckpoint(eager=self._eager)


class Show(Transformer):
    def __init__(
        self,
        *,
        n: int = 20,
        truncate: bool = True,
        vertical: bool = False,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
    ):
        """Print the first n rows to the console.

        Args:
            n (n | None):
                Number of rows to show.
                Defaults to 20.
            truncate (bool):
                If set to True, truncate strings longer than 20 chars by default.
                If set to a number greater than one, truncate long strings to
                length truncate and align cells right.
                Defaults to True.
            vertical (bool):
                If set to True, print output rows vertically (one line per
                column value). Defaults to False.
            columns (str | list(str) | None):
                A list of columns to select. Defaults to None.
            regex (str | None):
                Select the columns by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the columns by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
        """
        super().__init__()
        self._n: int = n
        self._truncate: bool = truncate
        self._vertical: bool = vertical
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )

    def _transform(self, df):
        selection: List[str] = self._get_selected_columns(df)
        if selection:
            df_show = df.select(selection)
        else:
            df_show = df

        df_show.show(n=self._n, truncate=self._truncate, vertical=self._vertical)
        return df


class ShowDistinct(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
        ascending: Union[bool, List[bool]] = True,
        limit: int = 100,
        truncate: bool = True,
        vertical: bool = False,
    ):
        """Show a DataFrame containing the distinct rows.

        Args:
            columns (str | list(str) | None):
                A list of columns to select for the 'distinct' operation.
                Defaults to None.
            ascending (bool | list(bool)):
                Boolean or list of boolean (default True).
                Sort ascending vs. descending.
                Specify a list for multiple sort orders.
                If a list is specified, the length of the list must
                equal the length of the cols.
            limit (int):
                Number of rows to show.
                Defaults to 100.
            truncate (bool):
                If set to True, truncate strings longer than 20 chars by default.
                If set to a number greater than one, truncate long strings to
                length truncate and align cells right.
                Defaults to True.
            vertical (bool):
                If set to True, print output rows vertically (one line per
                column value). Defaults to False.
        """
        super().__init__()
        self._columns: List[str] = ensure_flat_list(columns)
        self._limit: int = limit
        self._truncate: bool = truncate
        self._vertical: bool = vertical
        self._ascending: Union[bool, List[str]]

        if isinstance(ascending, (list, tuple)):
            if len(self._columns) != len(ascending):
                msg = 'If "ascending" is passed as <list>, it must have the '
                msg += 'same length of "columns". '
                msg += f"Found {len(self._columns)} instead of {len(ascending)}."
                raise AssertionError(msg)
        self._ascending = ascending

    def _sort_distinct(self, df):
        cols: List[str]
        if self._columns:
            cols = self._columns
            ret = df.select(self._columns)
        else:
            cols = df.columns
            ret = df

        return ret.distinct().sort(cols, ascending=self._ascending)

    def _transform(self, df):
        df_show = self._sort_distinct(df)
        df_show.show(n=self._limit, truncate=self._truncate, vertical=self._vertical)
        return df


class ToPandasToSpark(Transformer):
    def __init__(self):
        """Transform to Pandas and then revert to Spark.

        It can be used to truncate the logical plan of this DataFrame,
        which is especially useful in iterative algorithms where the plan
        may grow exponentially.
        """
        super().__init__()

    def _transform(self, df):
        return to_pandas_to_spark(df)
