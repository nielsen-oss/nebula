"""Transformers to assert certain conditions.

They don't manipulate the data but may trigger eager evaluation.
"""

import operator
from typing import Any, Hashable, Iterable, List, Optional, Union

from nlsn.nebula.auxiliaries import (
    assert_allowed,
    assert_is_integer,
    assert_only_one_non_none,
    ensure_flat_list,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.spark_util import cache_if_needed
from nlsn.nebula.storage import nebula_storage as ns

__all__ = [
    "AssertCount",
    "AssertNotEmpty",
    "AssertRowsAreDistinct",
    "CompareStoredValues",
    "DataFrameContainsColumns",
]


class AssertCount(Transformer):
    def __init__(
        self,
        *,
        number: Optional[int] = None,
        store_key: Optional[Hashable] = None,
        persist: bool = False,
        comparison: str = "eq",
    ):
        """Compare the number of rows with a reference value.

        Args:
            number (int | None):
                The expected count of rows in the DataFrame.
                If provided, 'store_key' should be None.
            store_key (int | None):
                The key to retrieve the expected count from nebula storage.
                If provided, 'number' should be None.
            persist (bool):
                If True, it caches the DataFrame before counting.
                Default to False
            comparison (str):
                The comparison operator to use for asserting the count.
                Valid values:
                    'eq': == (equal)
                    'ne': != (not equal)
                    'ge': >= (greater or equal)
                    'gt': > (greater than)
                    'le': <= (less equal)
                    'lt': < (less than)
                Default to 'eq' (equal)

        Raises:
        - AssertionError: If both 'number' and 'store_key' are provided or
            if neither is provided.
        - AssertionError: If 'comparison' is not a valid comparison operator.
        - ValueError: If the expected count is not an integer.
        """
        assert_only_one_non_none(number, store_key)
        self._d_sym = {
            "eq": "==",
            "ne": "!=",
            "ge": ">=",
            "gt": ">",
            "le": "<=",
            "lt": "<=",
        }
        comparison = comparison.strip().lower()
        assert_allowed(comparison, set(self._d_sym), "comparison")

        if number is not None:
            assert_is_integer(number, "number")

        super().__init__()
        self._n: Optional[int] = None if number is None else int(number)
        self._store_key: Optional[str] = store_key
        self._persist: bool = persist
        self._cmp: str = comparison

    def _transform(self, df):
        if self._n is None:
            self._n = ns.get(self._store_key)

        df = cache_if_needed(df, self._persist)

        count: int = df.count()

        cmp = getattr(operator, self._cmp)
        if not cmp(count, int(self._n)):
            sym: str = self._d_sym[self._cmp]
            msg = "AssertCount: DataFrame rows are not as expected. "
            msg += f"Actual count={count} | Expected count: {sym} {self._n}"
            raise AssertionError(msg)

        return df


class AssertNotEmpty(Transformer):
    def __init__(self, *, df_name: str = "Dataframe"):
        """Raise AssertionError if the input dataframe is empty.

        Args:
            df_name (str):
                DataFrame name to display if the assertion fails.
                The error message will be `df_name` + " is empty"
                Defaults to "Dataframe".

        Raises:
            AssertionError if the dataframe is empty.
        """
        super().__init__()
        self._df_name: str = df_name

    def _transform(self, df):
        if df.rdd.isEmpty():
            raise AssertionError(f"{self._df_name} is empty")
        return df


class AssertRowsAreDistinct(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
        persist: bool = False,
        raise_error: bool = True,
        log_level: str = "",
        log_message: str = "",
    ):
        """Raise AssertionError if there are duplicate rows in the specified column(s).

        The input dataframe remains untouched or just cached.

        Args:
            columns (str | list(str) | None):
                A list of columns to select. Defaults to None.
            regex (str | None):
                Selects the columns to select by using a regex
                pattern. Defaults to None.
            glob (str | None):
                Selects the columns to select by using a
                bash-like pattern. Defaults to None.
            startswith (str | iterable(str) | None):
                Select the columns whose names start with the
                provided string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select the columns whose names end with the
                provided string(s). Defaults to None.
            persist (bool):
                If True, cache the dataframe before asserting.
                Defaults to False.
            raise_error (bool):
                If True, raise an AssertionError.
                Defaults to True
            log_level (str):
                If provided, there must be a valid logger level like
                "debug", "info", "warning", etc.
                If not provided (default) it will not log in case of duplicates.
            log_message (str):
                Message to log in the case of duplicated rows.
                If provided, the `log_level` must be provided as well.

        Raises:
            AssertionError if duplicates in column(s) and `raise_error` is True.
        """
        if log_message and (not log_level):
            msg = "If `log_message` is provided, the `log_level` must be provided as well."
            raise AssertionError(msg)

        super().__init__()
        self._persist: bool = persist
        self._raise: bool = raise_error
        self._log_level: str = log_level
        self._log_msg: str = log_message
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
            df_dist = df.select(selection)
        else:
            df_dist = df

        df_dist = cache_if_needed(df_dist, self._persist)

        n_orig: int = df_dist.count()
        n_dist: int = df_dist.distinct().count()

        if n_orig != n_dist:
            if self._log_level and self._log_msg:
                log_func = getattr(logger, self._log_level)
                log_func(self._log_msg)

            if self._raise:
                msg = f"AssertDistinct: rows are not distinct: {n_orig} vs {n_dist}"
                raise AssertionError(msg)

        return df


class CompareStoredValues(Transformer):
    def __init__(
        self,
        *,
        key_a: Hashable,
        key_b: Optional[Hashable] = None,
        value: Optional[Any] = None,
        comparison: str,
    ):
        """Compare the value associated with the key 'key_a' with another value.

        'key_a' <compare> 'key_b' | 'value'

        Only one among 'key_b' or 'value' must be provided.

        Args:
            key_a (str):
                Storage key associated with the left member of the comparison.
            key_b (int | None):
                Storage key associated with the right member of the comparison.
                If provided, 'value' should be None.
            value (Any | None):
                The right member of the comparison.
                If provided, 'key_b' should be None.
            comparison (str):
                The comparison operator to use for asserting the count.
                Valid values:
                    'eq': == (equal)
                    'ne': != (not equal)
                    'ge': >= (greater or equal)
                    'gt': > (greater than)
                    'le': <= (less equal)
                    'lt': < (less than)
                    'isin': 'a' is in 'b'

        Raises:
        - AssertionError: If both 'value' and 'key_b' are provided or
            if neither is provided.
        - AssertionError: If 'comparison' is not a valid comparison operator.
        """
        assert_only_one_non_none(value, key_b)

        self._d_sym = {
            "eq": "==",
            "ne": "!=",
            "ge": ">=",
            "gt": ">",
            "le": "<=",
            "lt": "<=",
            "isin": "isin",
        }
        comparison = comparison.strip().lower()
        assert_allowed(comparison, set(self._d_sym), "comparison")

        super().__init__()
        self._key_a: Hashable = key_a
        self._key_b: Optional[Hashable] = key_b
        self._value: Optional[Any] = value
        self._cmp: str = comparison

    def _transform(self, df):
        a = ns.get(self._key_a)
        if self._key_b is not None:
            b = ns.get(self._key_b)
        else:
            b = self._value

        sym: str = self._d_sym[self._cmp]
        msg = f"CompareStoredValues: {a} {sym} {b}."

        if self._cmp == "isin":
            if a not in b:
                raise AssertionError(msg)
        else:
            cmp = getattr(operator, self._cmp)
            if not cmp(a, b):
                raise AssertionError(msg)

        return df


class DataFrameContainsColumns(Transformer):
    def __init__(self, *, columns: Union[str, List[str]]):
        """Raise AssertionError if the dataframe does not contain the expected columns.

        Args:
            columns (str | list(str)):
                The column or list of columns to check for existence.

        Raises:
            AssertionError if the dataframe does not contain the expected columns.
        """
        super().__init__()
        self._cols: List[str] = ensure_flat_list(columns)

    def _transform(self, df):
        actual_cols = set(df.columns)
        missing_cols = [i for i in self._cols if i not in actual_cols]

        if missing_cols:
            logger.info(f"Missing columns: {missing_cols}")
            raise AssertionError("Some expected columns are missing")

        return df
