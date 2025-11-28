"""Transformers for numerical operations."""

import operator as py_operator
from functools import reduce
from typing import Callable

import pyspark.sql.functions as F

from nlsn.nebula.base import Transformer

__all__ = [
    "MathOperator",
]


class MathOperator(Transformer):
    def __init__(
            self,
            *,
            strategy: dict | list[dict],
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
        self._strategy: list[dict] = strategy
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
            to_cast: str | None = el.get("cast")
            if to_cast:
                strat = strat.cast(to_cast)
            df = df.withColumn(el["new_column_name"], strat)
        return df
