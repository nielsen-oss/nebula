"""Transformers for numerical operations."""

import operator as py_operator
from functools import partial, reduce
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StructField

from nlsn.nebula.auxiliaries import (
    assert_allowed,
    assert_is_integer,
    assert_only_one_non_none,
    ensure_flat_list,
)
from nlsn.nebula.base import Transformer

__all__ = [
    "GroupedQuantileCut",
    "FloorOrCeil",
    "MathOperator",
]


class GroupedQuantileCut(Transformer):
    def __init__(
        self,
        *,
        quantiles: Union[int, List[float]],
        groupby_cols: Union[str, List[str]],
        input_col: str,
        output_col: Optional[str] = None,
        start_value: int = 0,
        fillna: int = 0,
        retbins: bool = False,
    ):
        """Quantile-based discretization function applied within groups.

        This transformer discretizes a numerical column into equal-sized
        buckets (quantiles) within each defined group. It leverages
        `pandas.qcut` through Spark's `applyInPandas` to perform
        the quantile-based categorization.

        For example, if you have 1000 values and specify 10 quantiles
        (deciles), each data point will be assigned to a category indicating
        its decile membership (e.g., 1st decile, 2nd decile, etc.).

        It is functionally similar to applying `pandas.qcut` on a grouped
        DataFrame using Spark's `applyInPandas`.
        https://pandas.pydata.org/docs/reference/api/pandas.qcut.html

        The `pandas.qcut` function is called with the following arguments for
        each group:

        ```python
        pandas.qcut(
            x, -> The column values within the current group
            quantiles, -> The 'quantiles' parameter passed to this transformer
            labels=False,
            retbins=False,
            precision=3,
            duplicates='drop',
            retbins=retbins,
        )
        ```

        The output column will contain either integer values representing the
        quantile bin for each row, or an array of bin edges, depending on the
        `retbins` parameter.

        Special handling for edge cases:
        - If a group contains only one unique value, all rows in that group
            will be assigned the `start_value` (if `retbins` is `False`).
        - If a group contains duplicate values such that `pandas.qcut` cannot
            uniquely assign a quantile (e.g., if `duplicates='drop'` is used
            and there are identical values at a quantile boundary), the
            resulting `NaN` values from `pandas.qcut` will be replaced by the
            integer specified in `fillna` (if `retbins` is `False`).

        The ranks (quantile bin values) will start from `start_value` if
        `retbins` is `False`.

        Args:
            quantiles (int | list(float)):
                Defines the quantiles to use for discretization.
                - If an `int`: The number of equal-sized quantiles
                  (e.g., 10 for deciles, 4 for quartiles).
                  Must be greater than 1.
                - If a `list` of `float`: An array of quantile probabilities
                  (e.g., `[0, .25, .5, .75, 1.]` for quartiles). All values
                   must be between 0.0 and 1.0 inclusive. If the leading `0.0`
                  or trailing `1.0` are omitted, they will be added automatically.
            groupby_cols (str | list(str)):
                The name(s) of the column(s) to group by before computing
                quantiles. The quantile calculation will be performed independently
                within each unique group.
            input_col (str):
                The name of the numerical column for which quantiles will be computed.
            output_col (str | None):
                The name of the output column that will store the computed
                quantile bin for each row or the array of bin edges if `retbins`
                is `True`. If not provided or `None`, the `input_col` will be
                overwritten with the results. Defaults to `None`.
            start_value (int):
                The integer value from which the quantile bins will start.
                For example, if `start_value` is 0, the bins will be 0, 1, 2, ...;
                if it's 1, they will be 1, 2, 3, ...
                This parameter is ignored if `retbins` is `True`. Defaults to 0.
            fillna (int):
                The integer value used to replace `NaN` results from `pandas.qcut`.
                This typically occurs when `duplicates='drop'` is used and
                there are identical values at a quantile boundary, or if a
                group has not enough unique values to form the requested
                quantiles. This parameter is ignored if `retbins` is `True`.
                Defaults to 0.
            retbins (bool):
                If `True`, the transformer returns the bin edges of the quantiles
                instead of the quantile bin for each row. The output column
                will be an `array<double>` representing the bin edges. The length
                of this array will be `len(quantiles) + 1` (if `quantiles` is a
                list) or `quantiles + 1` (if `quantiles` is an int).
                If `True`, `fillna` and `start_value` parameters are ignored
                and must be left at their default values. Defaults to `False`.
        """
        if not groupby_cols:
            raise ValueError("At least one groupby column must be provided.")

        if retbins and (fillna or start_value):
            raise ValueError("'retbins' cannot be used with 'fillna' or 'start_value'")

        super().__init__()
        assert_is_integer(fillna, "fillna")
        assert_is_integer(start_value, "start_value")
        self._fillna: int = int(fillna)
        self._start: int = int(start_value)
        self._retbins: bool = retbins

        self._q = Union[int, List[float]]
        if isinstance(quantiles, (int, float, str)):
            assert_is_integer(quantiles, "quantiles")
            if quantiles < 2:
                raise ValueError(
                    f"If 'quantiles' is an integer, it must be > 1. Provided {quantiles}"
                )
            self._q = quantiles
        else:
            quantiles = ensure_flat_list(quantiles)
            if not all(isinstance(q, (float, int)) for q in quantiles):
                # Keep any, as 0 and 1 can be seen as int
                raise ValueError("If quantiles is a list, it must be a list of floats.")
            if not all(0.0 <= q <= 1.0 for q in quantiles):
                raise ValueError(
                    "All 'quantiles' values must be between 0.0 and 1.0 inclusive."
                )
            self._q = sorted(quantiles[:])
            if self._q[0] != 0.0:
                self._q.insert(0, 0.0)
            if self._q[-1] != 1.0:
                self._q.append(1.0)

        self._groupby: List[str] = ensure_flat_list(groupby_cols)
        self._input_col: str = input_col
        self._output_col: str = output_col or input_col

    def _transform(self, df):
        schema = df.schema[:]  # Don't pass the reference, make a copy
        if self._retbins:
            schema.add(StructField("_qcut_", ArrayType(DoubleType()), True))
        else:
            schema.add(StructField("_qcut_", IntegerType(), True))

        def _func(
            _df, *, _col: str, _q, _fillna: int, _start: int, _retbins: bool
        ):  # pragma: no cover
            # _values = np.unique(df["duration"].values)  # already sorted
            _values = _df[_col].values
            _n = len(_values)
            if _retbins:
                if _n <= 1:
                    _df["_qcut_"] = [[]] * _n
                    return _df
                _, _ar_rank = pd.qcut(
                    _values, _q, labels=False, duplicates="drop", retbins=True
                )
                _li_rank = _ar_rank.tolist()
                _df["_qcut_"] = [_li_rank] * _n
            else:
                if _n <= 1:
                    _df["_qcut_"] = np.int32(_start)
                    return _df
                _ar_rank = (
                    pd.qcut(_values, _q, labels=False, duplicates="drop") + _start
                )
                _ar_rank[np.isnan(_ar_rank)] = _fillna
                _df["_qcut_"] = _ar_rank.astype(np.int32)
            return _df

        func = partial(
            _func,
            _col=self._input_col,
            _q=self._q,
            _fillna=self._fillna,
            _start=self._start,
            _retbins=self._retbins,
        )

        ret = df.groupby(self._groupby).applyInPandas(func, schema=schema)

        if self._output_col in df.columns:
            ret = ret.drop(self._output_col)

        ret = ret.withColumnRenamed("_qcut_", self._output_col)
        return ret


class FloorOrCeil(Transformer):
    def __init__(
        self,
        *,
        operation: Optional[str] = None,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
        input_output_columns: Optional[Dict[str, str]] = None,
    ):
        """Perform 'floor' or 'ceil' operations on specified columns.

        Only one argument among:
        - 'columns'
        - 'regex'
        - 'glob'
        - 'input_output_columns'
        can be used to select the columns on which to perform the operation.
        The only argument that allows the user to create new fields with the
        ceil/floor values is 'input_output_columns'. With all the other
        arguments, the transformer replaces the new values in the input columns.

        Args:
            operation (str):
                The operation to perform, either 'floor' or 'ceil'.
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
            input_output_columns (dict(str, str) | None):
                If provided, a dictionary where the keys (str) represent the
                input columns and the values (str) represent the output columns.
                Defaults to None.
        """
        assert_allowed(operation, {"floor", "ceil"}, operation)

        # Assert only one input columns selection strategy is provided.
        assert_only_one_non_none(columns, regex, glob, input_output_columns)

        if input_output_columns:
            if not isinstance(input_output_columns, dict):
                raise TypeError('"input_output_columns" must be <dict>')
            if not all(isinstance(i, str) for i in input_output_columns):
                raise TypeError('"input_output_columns" keys must be <str>')
            values = input_output_columns.values()
            if not all(isinstance(i, str) for i in values):
                raise TypeError('"input_output_columns" values must be <str>')
            if len(values) != len(set(values)):
                raise ValueError('Duplicated values in "input_output_columns"')

        super().__init__()
        self._operation: str = operation
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )
        self._input_output_columns: Optional[Dict[str, str]] = input_output_columns

    def _transform(self, df):
        op: F.col = getattr(F, self._operation)
        if self._input_output_columns:
            set_new_cols = set(self._input_output_columns.values())
            new_cols = [op(k).alias(v) for k, v in self._input_output_columns.items()]
            old_cols = [i for i in df.columns if i not in set_new_cols]
            out_cols = old_cols + new_cols
        else:
            selection: List[str] = self._get_selected_columns(df)
            out_cols = [op(i).alias(i) if i in selection else i for i in df.columns]

        return df.select(*out_cols)


class MathOperator(Transformer):
    def __init__(
        self,
        *,
        strategy: Union[dict, List[dict]],
    ):
        """Apply a mathematical operator to columns and constants.

        Args:
            strategy (dict | list(dict)):
                Strategy to create the new column.
                This parameter is a list of dictionaries like:
                [
                    {
                        'new_column_name': <(str) Output column name>,
                        'cast': <(str | None) If provided, the data type to cast the
                            output column to>,
                        'strategy': <(list(dict)) list of dictionaries like: [
                                {'column': 'col1'},
                                {'column': 'col2', 'cast': 'double'}
                                {'constant': 4.2},
                            ],
                            where each dictionary represents a member of the
                            mathematical operation and the allowed keys are:
                            - 'column': to select a dataframe column
                            - 'constant': to use a literal value instead of a
                                dataframe column. 'column' and 'constant' are
                                mutually exclusive.
                            - 'cast': (Optional) Cast the column or constant
                                before performing the mathematical operations.
                        'operations': <(list(str)) Operations to execute
                            sequentially from left to right, following the
                            order of 'strategy'>
                    },
                ]

        Example:
        strategy_to_cols = [
            {
                'new_column_name': 'col4',
                'cast': 'long',
                'strategy': [
                    {'column': 'col1'},
                    {'column': 'col2', 'cast': 'double'}
                    {'constant': 4.2},
                ],
                'operations': <(list(str)) operation to execute sequentially with
                    the same order of the 'strategy'>
            },
        ]

        1. Add 'col1' to 'col2' (after having cast 'col2' to double)
        2. Take the previous result and subtract the literal value 4.2
        3. Create the new column 'col4'
        4. Cast 'col4' to long
        """
        if not isinstance(strategy, (list, tuple, dict)):
            raise TypeError('"strategy" must be <list> or <dict>')

        if isinstance(strategy, dict):
            strategy = [strategy]

        super().__init__()
        self._strategy: List[dict] = strategy
        self._operators_map: dict = {
            "add": py_operator.add,
            "sub": py_operator.sub,
            "mul": py_operator.mul,
            "div": py_operator.truediv,
            "pow": py_operator.pow,
        }

    @staticmethod
    def _get_constant_or_col(o: dict) -> F.col:
        n = len(o)
        if n >= 3:
            raise ValueError(f"Dict length must be 1 or 2, found {o}")

        # 'o' must have 'column' or 'constant' as a key but not both at the same time
        flag_constant_or_column = ("column" in o) + ("constant" in o)

        if flag_constant_or_column != 1:
            msg = "Strategy lists should contain only column or constant as "
            msg += "a mandatory key and the keys are mutually exclusive, "
            msg += f"the key cast is optional found {o.keys()}"
            raise ValueError(msg)

        cast = o.get("cast")
        col_name = o.get("column")
        if col_name is not None:
            spark_col = F.col(col_name)
        else:
            spark_col = F.lit(o["constant"])

        return spark_col.cast(cast) if cast else spark_col

    def _get_op(self, op) -> Callable:
        ops = set(self._operators_map)
        if op not in ops:
            raise ValueError(f"Operator must be in {ops}, found {op}")
        return self._operators_map[op]

    def _get_aggregation_list(self, strat_dict) -> F.col:
        strategy = strat_dict["strategy"]
        operators = strat_dict["operations"]
        if len(strategy) - 1 != len(operators):
            msg = "len of Strategy must be len of operators -1, found "
            msg += f"{len(strategy), len(operators)}"
            raise ValueError(msg)

        l1 = [self._get_constant_or_col(i) for i in strategy]
        ops = iter([self._get_op(op) for op in operators])
        col_list = reduce(lambda a, b: next(ops)(a, b), l1)
        return col_list

    def _transform(self, df):
        el: dict
        for el in self._strategy:
            strat: F.col = self._get_aggregation_list(el)
            to_cast: Optional[str] = el.get("cast")
            if to_cast:
                strat = strat.cast(to_cast)
            df = df.withColumn(el["new_column_name"], strat)
        return df
