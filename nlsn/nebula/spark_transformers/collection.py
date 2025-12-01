"""General Purposes Transformers."""

from functools import reduce
from typing import Any

from pyspark.sql import functions as F

from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_util import (
    ensure_spark_condition,
    get_spark_condition,
)

__all__ = [
    "When",
]


class When(Transformer):
    def __init__(
            self,
            *,
            output_column: str,
            conditions: list[dict],
            otherwise_constant: Any = None,
            otherwise_column: str | None = None,
            cast_output: str | None = None,
    ):
        """Apply conditional logic to create a column based on specified conditions.

        Args:
            output_column (str):
                The name of the output column to store the results.
            conditions (list(dict)):
                A list of dictionaries specifying the conditions and their
                corresponding outputs.
                Each dictionary should have the following keys:
                    - 'input_col' (str): The name of the input column.
                    - 'operator' (str): The operator to use for the condition.
                        Refer Filter Transformer docstring
                    - 'value' (Any): Value to compare against.
                    - 'comparison_column' (str): name of column to be compared
                        with `input_col`
                    - 'output_constant' (Any): The output constant value if the
                        condition is satisfied.
                    - 'output_column' (str): The column name if the condition is
                        satisfied.

                NOTE: Either `value` or `comparison_column` (not both) must be
                    provided for python operators that require a comparison value
                NOTE: Either `output_constant` or `output_column` (not both) must
                    be provided. If both are provided, `output_column` will
                    always supersede `output_constant`

            otherwise_constant (object):
                The output constant if none of the conditions are satisfied.
            otherwise_column (str):
                The column if none of the conditions are satisfied.
            cast_output (str | None):
                Datatype of output column

        NOTE: Either `otherwise_constant` or `otherwise_column` (not both) must
        be provided. If both are provided, `otherwise_column` will always
        supersede `otherwise_constant`

        Raises:
            AssertionError: If the 'ne' (not equal) operator is used in the conditions.

        Examples:
            - transformer: When
              params:
                output_column: "p_out"
                conditions:
                  - {"input_col": primary, "operator": eq, "value": 1, output_column: col_1}
                  - {"input_col": primary, "operator": eq, "value": 0, output_column: col_2}
                  - {"input_col": primary, "operator": eq, "value": 0, output_constant: 999.0}
                otherwise_column: "another_col"
                cast_output: "double"
        """
        super().__init__()
        self._output_column: str = output_column

        self._otherwise: F.col
        if otherwise_column:
            self._otherwise: F.col = F.col(otherwise_column)
        else:
            self._otherwise: F.col = F.lit(otherwise_constant)

        if cast_output:
            self._otherwise = self._otherwise.cast(cast_output)

        for el in conditions:
            operator = el.get("operator")
            value = el.get("value")
            compare_col = el.get("comparison_column")
            ensure_spark_condition(operator, value, compare_col)

        self._conditions: list[dict[str, Any]] = conditions
        self._cast_output: str | None = cast_output

    def _transform(self, df):
        spark_conds = []
        for el in self._conditions:
            cond = get_spark_condition(
                df,
                el.get("input_col"),
                el.get("operator"),
                value=el.get("value"),
                compare_col=el.get("comparison_column"),
            )
            out_col = el.get("output_column")
            out = F.col(out_col) if out_col else F.lit(el.get("output_constant"))
            if self._cast_output:
                out = out.cast(self._cast_output)
            spark_conds.append((cond, out))

        out_col = reduce(lambda f, y: f.when(*y), spark_conds, F).otherwise(
            self._otherwise
        )
        return df.withColumn(self._output_column, out_col)
