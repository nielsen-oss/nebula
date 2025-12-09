import operator as py_operator
from typing import Callable, Any

import narwhals as nw

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.base import Transformer
from nlsn.nebula.nw_util import validate_operation, get_condition
from nlsn.nebula.transformers._constants import NW_TYPES

__all__ = [
    "InjectData",
    "MathOperator",
    "When",
]


class InjectData(Transformer):  # FIXME: move to keyword. add kwargs

    def __init__(
            self,
            *,
            data: dict | list,
            storage_key: str,
            broadcast: bool = False,
    ):
        """Temporary: Will become pipeline keyword post-migration.

        Creates a DataFrame from provided data (typically Jinja-templated
        values) and stores it for later use in joins or other operations.
        The input DataFrame passes through unchanged.

        Example:
            # In Jinja-templated YAML:
            - transformer: InjectData
              params:
                storage_key: "run_context"
                data:
                  run_date: ["{{ run_date }}"]
                  customer: ["{{ customer_id }}"]

            # Later in pipeline:
            - transformer: Join
              params:
                table: "run_context"
                on: [customer]
        """
        super().__init__()
        self._data = data or []
        self._storage_key = storage_key
        self._broadcast = broadcast

    def _post(self, df_in, df_out):
        # Match input type
        if isinstance(df_in, (nw.DataFrame, nw.LazyFrame)):
            df_out = nw.from_native(df_out)

        ns.set(self._storage_key, df_out)
        return df_in  # Pass-through

    def _transform_pandas(self, df):
        import pandas as pd
        ret = pd.DataFrame(self._data)
        return self._post(df, ret)

    def _transform_polars(self, df):
        import polars as pl
        ret = pl.DataFrame(self._data)
        return self._post(df, ret)

    def _transform_spark(self, df):
        from nlsn.nebula.spark_util import get_spark_session
        import pandas as pd

        ss = get_spark_session(df)
        df_pd = pd.DataFrame(self._data)  # FIXME: spark has it own methods to create dfs
        ret = ss.createDataFrame(df_pd)
        if self._broadcast:  #
            from pyspark.sql.functions import broadcast

            ret = broadcast(ret)
        return self._post(df, ret)


class MathOperator(Transformer):
    """Apply mathematical operators to columns and constants.

    This transformer enables declarative mathematical expressions by
    applying a sequence of operations (add, sub, mul, div, pow) to
    columns and/or constant values.

    Examples:
        >>> # Simple addition: result = col1 + col2
        >>> MathOperator(strategy={
        ...     'new_column_name': 'result',
        ...     'strategy': [
        ...         {'column': 'col1'},
        ...         {'column': 'col2'}
        ...     ],
        ...     'operations': ['add']
        ... })

        >>> # Complex: total = (price * quantity) - discount
        >>> MathOperator(strategy={
        ...     'new_column_name': 'total',
        ...     'strategy': [
        ...         {'column': 'price', 'cast': 'double'},
        ...         {'column': 'quantity'},
        ...         {'column': 'discount'}
        ...     ],
        ...     'operations': ['mul', 'sub']
        ... })

        >>> # With constants: normalized = (value - 100) / 50
        >>> MathOperator(strategy={
        ...     'new_column_name': 'normalized',
        ...     'cast': 'float',
        ...     'strategy': [
        ...         {'column': 'value'},
        ...         {'constant': 100},
        ...         {'constant': 50}
        ...     ],
        ...     'operations': ['sub', 'div']
        ... })

        >>> # Multiple columns at once
        >>> MathOperator(strategy=[
        ...     {
        ...         'new_column_name': 'total_price',
        ...         'strategy': [
        ...             {'column': 'unit_price'},
        ...             {'column': 'quantity'}
        ...         ],
        ...         'operations': ['mul']
        ...     },
        ...     {
        ...         'new_column_name': 'price_with_tax',
        ...         'strategy': [
        ...             {'column': 'total_price'},
        ...             {'constant': 1.2}
        ...         ],
        ...         'operations': ['mul']
        ...     }
        ... ])
    """

    def __init__(
            self,
            *,
            strategy: dict | list[dict],
    ):
        """Initialize MathOperator transformer.

        Args:
            strategy: Single dict or list of dicts defining operations.
                Each dict must contain:
                - new_column_name (str): Name of output column
                - strategy (list[dict]): Operands in order, each with:
                    - column (str): Column name (mutually exclusive with constant)
                    - constant: Literal value (mutually exclusive with column)
                    - cast (str | None): Optional type for this operand
                - operations (list[str]): Operations applied left-to-right.
                    Must be one of: 'add', 'sub', 'mul', 'div', 'pow'
                - cast (str | None): Optional final output type

        Raises:
            TypeError: If strategy is not dict or list.
            ValueError: If strategy structure is invalid.
        """
        if isinstance(strategy, dict):
            strategy = [strategy]
        elif not isinstance(strategy, (list, tuple)):
            raise TypeError(
                f'"strategy" must be dict or list, found {type(strategy).__name__}'
            )

        super().__init__()
        self._strategy: list[dict] = strategy
        self._operators_map: dict[str, Callable] = {
            "add": py_operator.add,
            "sub": py_operator.sub,
            "mul": py_operator.mul,
            "div": py_operator.truediv,
            "pow": py_operator.pow,
        }

    @staticmethod
    def _cast_type(dtype_str: str) -> nw.dtypes.DType:
        """Convert string type name to narwhals dtype.

        Args:
            dtype_str: Type name (e.g., 'int', 'double', 'string')

        Returns:
            Narwhals dtype object

        Raises:
            ValueError: If type name is not recognized
        """
        dtype_lower = dtype_str.lower()
        if dtype_lower not in NW_TYPES:
            raise ValueError(
                f"Unknown type '{dtype_str}'. "
                f"Must be one of: {sorted(NW_TYPES.keys())}"
            )
        return NW_TYPES[dtype_lower]

    def _get_constant_or_col(self, operand: dict):
        """Convert strategy operand dict to narwhals expression.

        Args:
            operand: Dict with 'column' or 'constant' key, and optional 'cast'

        Returns:
            Narwhals expression

        Raises:
            ValueError: If operand dict structure is invalid
        """
        if len(operand) > 2:
            raise ValueError(
                f"Operand dict can have at most 2 keys (column/constant + cast), "
                f"found {len(operand)}: {operand}"
            )

        has_col = "column" in operand
        has_const = "constant" in operand

        if has_col == has_const:  # Both True or both False
            raise ValueError(
                f"Must specify exactly one of 'column' or 'constant'. "
                f"Found keys: {list(operand.keys())}"
            )

        # Create base expression
        if has_col:
            expr = nw.col(operand["column"])
        else:
            expr = nw.lit(operand["constant"])

        # Apply cast if specified
        cast_to = operand.get("cast")
        if cast_to:
            expr = expr.cast(self._cast_type(cast_to))

        return expr

    def _get_op(self, op_name: str) -> Callable:
        """Get operator function by name.

        Args:
            op_name: Operator name

        Returns:
            Operator function

        Raises:
            ValueError: If operator name is not recognized
        """
        if op_name not in self._operators_map:
            raise ValueError(
                f"Operator must be one of {set(self._operators_map.keys())}, "
                f"found '{op_name}'"
            )
        return self._operators_map[op_name]

    def _build_expression(self, strat_dict: dict):
        """Build narwhals expression from strategy dict.

        Args:
            strat_dict: Strategy dict with 'strategy' and 'operations' keys

        Returns:
            Narwhals expression

        Raises:
            ValueError: If lengths don't match or structure is invalid
        """
        strategy = strat_dict["strategy"]
        operations = strat_dict["operations"]

        # Validate lengths
        if len(strategy) - 1 != len(operations):
            raise ValueError(
                f"Strategy must have exactly one more element than operations. "
                f"Found strategy length={len(strategy)}, "
                f"operations length={len(operations)}"
            )

        # Convert all operands to narwhals expressions
        exprs = [self._get_constant_or_col(item) for item in strategy]

        # Apply operations left-to-right using reduce
        op_funcs = [self._get_op(op_name) for op_name in operations]

        result = exprs[0]
        for op_func, next_expr in zip(op_funcs, exprs[1:]):
            result = op_func(result, next_expr)

        return result

    def _transform_nw(self, df):
        """Apply mathematical operations to create new columns.

        Args:
            df: Narwhals DataFrame or LazyFrame

        Returns:
            DataFrame/LazyFrame with new columns added
        """
        new_cols = []

        for strat_dict in self._strategy:
            # Build the expression
            expr = self._build_expression(strat_dict)

            # Apply final cast if specified
            final_cast = strat_dict.get("cast")
            if final_cast:
                expr = expr.cast(self._cast_type(final_cast))

            # Alias with the new column name
            new_cols.append(expr.alias(strat_dict["new_column_name"]))

        return df.with_columns(new_cols)


class When(Transformer):
    """Create a new column using conditional logic (if-then-else).

    Apply a chain of conditions to determine the output value for each row.
    Conditions are evaluated in order, and the first matching condition's output
    is used. If no conditions match, the 'otherwise' value is used.

    This is the Narwhals equivalent of SQL CASE WHEN or Spark's F.when().
    """

    def __init__(
            self,
            *,
            output_col: str,
            conditions: list[dict[str, Any]],
            otherwise_constant: Any = None,
            otherwise_col: str | None = None,
            cast_output: str | None = None,
    ):
        """Initialize When transformer.

        Args:
            output_col: Name of the output column to create.

            conditions: List of condition dictionaries. Each dictionary specifies:
                - 'input_col' (str): Column to evaluate the condition on
                - 'operator' (str): Comparison operator (same as Filter operators)
                - 'value' (Any, optional): Value to compare against
                - 'compare_col' (str, optional): Column to compare against
                - 'output_constant' (Any, optional): Value to output if condition matches
                - 'output_col' (str, optional): Column to output if condition matches

                **Important:** Either 'value' or 'compare_col' must be provided
                (not both) for operators that need comparison.

                **Important:** Either 'output_constant' or 'output_col' must be
                provided (not both). If both are provided, 'output_col' takes precedence.

            otherwise_constant: Default value if no conditions match.
                Cannot be used with otherwise_col.

            otherwise_col: Default column if no conditions match.
                Cannot be used with otherwise_constant. If both are provided,
                otherwise_col takes precedence.

            cast_output: Cast the output column to this dtype (e.g., "Int64", "Float64",
                "String"). Applied to all outputs (condition outputs and otherwise value).

        Raises:
            ValueError: If conditions have invalid operators or parameter combinations.
            TypeError: If condition values have wrong types.

        Notes:
            **Condition Evaluation Order:**
            Conditions are evaluated in the order provided. The first matching
            condition determines the output. Subsequent conditions are not evaluated
            for rows that already matched.

            **Null Handling:**
            - If a condition evaluates to null (e.g., null > 5 → null), it's treated
              as False (condition doesn't match).
            - The otherwise value is used for rows where no conditions match,
              including rows where all conditions evaluated to null.

            **Operator Support:**
            Supports all operators from Filter:
            - Standard: eq, ne, lt, le, gt, ge
            - Null/NaN: is_null, is_not_null, is_nan, is_not_nan
            - String: contains, starts_with, ends_with
            - Set: is_in, is_not_in
            - Range: is_between

            See Filter documentation for detailed operator behavior.

            **Type Casting:**
            If cast_output is specified, all outputs (from conditions and otherwise)
            are cast to that type. This ensures consistent output types across all
            branches, which is important for strongly-typed backends like Polars.

        Examples:
            Simple categorization:
                >>> When(
                ...     output_col="age_group",
                ...     conditions=[
                ...         {"input_col": "age", "operator": "lt",
                ...          "value": 18, "output_constant": "minor"},
                ...         {"input_col": "age", "operator": "lt",
                ...          "value": 65, "output_constant": "adult"},
                ...     ],
                ...     otherwise_constant="senior"
                ... )

            Using column outputs:
                >>> When(
                ...     output_col="best_score",
                ...     conditions=[
                ...         {"input_col": "score_a", "operator": "gt",
                ...          "compare_col": "score_b", "output_col": "score_a"},
                ...     ],
                ...     otherwise_col="score_b"
                ... )

            Multiple conditions with type casting:
                >>> When(
                ...     output_col="status_code",
                ...     conditions=[
                ...         {"input_col": "status", "operator": "eq",
                ...          "value": "active", "output_constant": 1},
                ...         {"input_col": "status", "operator": "eq",
                ...          "value": "pending", "output_constant": 2},
                ...         {"input_col": "status", "operator": "is_null",
                ...          "output_constant": -1},
                ...     ],
                ...     otherwise_constant=0,
                ...     cast_output="int64"
                ... )

            String operations:
                >>> When(
                ...     output_col="email_domain",
                ...     conditions=[
                ...         {"input_col": "email", "operator": "contains",
                ...          "value": "@company.com", "output_constant": "internal"},
                ...         {"input_col": "email", "operator": "contains",
                ...          "value": "@", "output_constant": "external"},
                ...     ],
                ...     otherwise_constant="invalid"
                ... )

            Set membership:
                >>> When(
                ...     output_col="priority",
                ...     conditions=[
                ...         {"input_col": "user_id", "operator": "is_in",
                ...          "value": [1, 2, 3], "output_constant": "high"},
                ...         {"input_col": "user_id", "operator": "is_in",
                ...          "value": [4, 5, 6], "output_constant": "medium"},
                ...     ],
                ...     otherwise_constant="low"
                ... )

            Config-driven (YAML):
                - transformer: When
                  params:
                    output_col: risk_level
                    conditions:
                      - input_col: amount
                        operator: gt
                        value: 10000
                        output_constant: high
                      - input_col: amount
                        operator: gt
                        value: 1000
                        output_constant: medium
                    otherwise_constant: low
                    cast_output: string

        See Also:
            - Filter: For row filtering using the same condition operators
            - FillNa: For simpler null value replacement
            - Coalesce: For selecting first non-null value from multiple columns
        """
        super().__init__()

        # Validate all conditions upfront
        for i, cond in enumerate(conditions):
            operator = cond.get("operator")
            value = cond.get("value")
            compare_col = cond.get("compare_col") or cond.get("comparison_column")

            # Validate the condition parameters
            validate_operation(operator, value=value, compare_col=compare_col)

            # Ensure output is specified
            has_output_constant = "output_constant" in cond
            has_output_col = "output_col" in cond or "output_column" in cond

            if not (has_output_constant or has_output_col):
                raise ValueError(
                    f"Condition {i} must specify either 'output_constant' or 'output_col'"
                )

        # Validate otherwise clause
        if otherwise_constant is None and otherwise_col is None:
            raise ValueError(
                "Must specify either 'otherwise_constant' or 'otherwise_col'"
            )

        self._output_col: str = output_col
        self._conditions = conditions
        self._otherwise_constant = otherwise_constant
        self._otherwise_col: str | None = otherwise_col
        self._cast_output = NW_TYPES.get(cast_output)

    def _transform_nw(self, df):
        """Build the when-then-else expression using Narwhals.

        Narwhals uses nested when expressions rather than chained ones.
        We build from the inside out (reversed conditions) so that the
        first condition in the list is evaluated first.
        """

        # Start with the otherwise clause (innermost)
        if self._otherwise_col:
            result_expr = nw.col(self._otherwise_col)
        else:
            result_expr = nw.lit(self._otherwise_constant)

        # Cast if specified
        if self._cast_output:
            result_expr = result_expr.cast(self._cast_output)

        # Build nested when-then-otherwise expressions
        # Reverse so first condition in list is checked first
        for cond in reversed(self._conditions):
            # Extract condition parameters (support both naming conventions)
            input_col = cond["input_col"]
            operator = cond["operator"]
            value = cond.get("value")
            compare_col = cond.get("compare_col") or cond.get("comparison_column")

            # Build the condition
            condition = get_condition(
                input_col,
                operator,
                value=value,
                compare_col=compare_col,
            )

            # Determine the output (support both naming conventions)
            output_col = cond.get("output_col") or cond.get("output_column")
            if output_col:
                output_expr = nw.col(output_col)
            else:
                output_expr = nw.lit(cond["output_constant"])

            # Cast if specified
            if self._cast_output:
                output_expr = output_expr.cast(self._cast_output)

            # Nest the when-then-otherwise
            result_expr = nw.when(condition).then(output_expr).otherwise(result_expr)

        return df.with_columns(result_expr.alias(self._output_col))
