"""MapType Columns Manipulations."""

import operator
from functools import reduce
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from pyspark.sql import functions as F
from pyspark.sql.types import MapType

from nlsn.nebula.auxiliaries import (
    assert_allowed,
    check_if_columns_are_present,
    ensure_flat_list,
    ensure_nested_length,
    is_list_uniform,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.spark_util import null_cond_to_false

__all__ = [
    "ColumnsToMap",
    "KeysToArray",
    "MapToColumns",
]


class ColumnsToMap(Transformer):
    def __init__(
        self,
        *,
        output_column: str,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
        exclude_columns: Optional[Union[str, Iterable[str]]] = None,
        cast_values: Optional[str] = None,
        drop_input_columns: bool = False,
    ):
        """Create a MapType field using the provided columns.

        Args:
            output_column (str):
                Name of the output column.
            columns (str | list(str) | None):
                A list of columns to select. Defaults to None.
            regex (str | None):
                Take the columns to select by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Take the columns to select by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
            exclude_columns (str | iterable(str) | None):
                List of columns that will not be selected. Defaults to None.
            cast_values (str | DataType | None):
                If provided, cast the values to the specified type.
                Defaults to None.
            drop_input_columns (bool)
                If True, drop the input columns. Defaults to False.
        """
        super().__init__()
        self._output_column = output_column
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )
        self._excl_cols: List[str] = ensure_flat_list(exclude_columns)
        self._cast_value: Optional[str] = cast_values
        self._drop: bool = drop_input_columns

    def _transform(self, df):
        input_columns: List[str] = self._get_selected_columns(df)
        if self._excl_cols:
            input_columns = [i for i in input_columns if i not in self._excl_cols]

        pairs = []
        for name in input_columns:
            key = F.lit(name)
            value = F.col(name)
            if self._cast_value is not None:
                value = value.cast(self._cast_value)
            pairs.append((key, value))

        out = F.create_map(*chain.from_iterable(pairs))
        df = df.withColumn(self._output_column, out)

        if self._drop:
            return df.drop(*input_columns)
        return df



class KeysToArray(Transformer):
    def __init__(
        self,
        *,
        input_column: str,
        output_column: Optional[str] = None,
        sort: bool = False,
    ):
        """Extract keys from a MapType column to create an ArrayType field.

        Args:
            input_column (str):
                Name of the column containing the MapType.
            output_column (str, | None):
                If not provided, output is stored in the <input_col>.
            sort (bool):
                If True, sort the output array. Defaults to False.

        Raises:
            ValueError: If the specified input column does not exist.
            TypeError: If the input column is not a MapType.
        """
        super().__init__()
        self._input_column: str = input_column
        self._output_column: str = output_column if output_column else input_column
        self._sort: bool = sort

    def _transform(self, df):
        check_if_columns_are_present([self._input_column], df.columns)
        col_data_type = df.schema[self._input_column].dataType
        if not isinstance(col_data_type, MapType):
            raise TypeError(f"Column '{self._input_column}' is not of MapType.")

        func = F.map_keys(self._input_column)
        if self._sort:
            func = F.sort_array(func)

        return df.withColumn(self._output_column, func)


class MapToColumns(Transformer):
    def __init__(
        self,
        *,
        input_column: str,
        output_columns: Union[List[str], List[List[str]], Dict[Any, str]],
    ):
        """Extract keys from a MapType column and create new columns.

        Args:
            input_column (str):
                The name of the MapType column to extract keys from.
            output_columns (str, | None):
                Keys to extract to create new columns. It can be a:
                - list of strings
                - Nested list of 2-element list, where the
                    - 1st value represents the key to extract.
                    - 2nd value represents the alias for the new column.
                        It must be a string.
                - Dictionary where:
                    - the key represents the MapType key to extract
                    - the value represents the alias for the new column.
                        It must be a string.

        Raises (ValueError):
            AssertionError: If 'output_columns' is an empty list / dictionary.
            TypeError: If the input column is not a MapType.
            TypeError: If the 'output_columns' is not specified within a
                valid list or dictionary format.
        """
        if not output_columns:
            raise AssertionError("'output_columns' cannot be empty")

        super().__init__()
        self._input_col: str = input_column
        self.cols: List[Tuple[str, str]]

        values: list  # A list created for checks only

        if isinstance(output_columns, (list, tuple)):
            if is_list_uniform(output_columns, str):
                self.cols = [(i, i) for i in output_columns]
                # Do not perform further checks
                return
            else:
                # Try to convert to dict to check if the list is made of
                # 2-element iterables and ensure the first element is hashable
                if not ensure_nested_length(output_columns, 2):
                    msg = "If 'output_columns' is provided as nested list "
                    msg += "all the sub-lists must have length equal to 2."
                    raise TypeError(msg)
                d = dict(output_columns)
                # Extract values to perform some checks later
                values = list(d.values())
                self.cols = output_columns

        elif isinstance(output_columns, dict):
            values = list(output_columns.values())
            self.cols = list(output_columns.items())

        else:
            msg = "'output columns' must be a (list(str)) or a "
            msg += "(list(tuple(str, str))) or a (dict(str, str))"
            raise TypeError(msg)

        if not is_list_uniform(values, str):
            msg = "All values provided for column aliases must be <string>"
            raise TypeError(msg)

        if len(set(values)) != len(output_columns):
            msg = "All values provided for column aliases must not contain duplicates"
            raise TypeError(msg)

    def _transform(self, df):
        col_data_type = df.schema[self._input_col].dataType
        if not isinstance(col_data_type, MapType):
            raise TypeError(f"Column '{self._input_col}' is not of MapType.")

        items = [F.col(self._input_col).getItem(i).alias(j) for i, j in self.cols]
        return df.select(*df.columns, *items)
