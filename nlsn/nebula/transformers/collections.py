import operator as py_operator
from typing import Iterable, Literal, Callable, Any

import narwhals as nw

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.auxiliaries import (
    assert_allowed,
    assert_at_least_one_non_null,
    ensure_flat_list,
    assert_only_one_non_none,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.df_types import get_dataframe_type
from nlsn.nebula.nw_util import null_cond_to_false, validate_operation, get_condition
from nlsn.nebula.transformers._constants import NW_TYPES

__all__ = [
    "AppendDataFrame",
    "DropNulls",
    "Filter",
    "GroupBy",
    "Join",
    "InjectData",
    "MathOperator",
    "Pivot",
    "Unpivot",
    "When",
]


class AppendDataFrame(Transformer):
    def __init__(
            self,
            *,
            store_key: str | None = None,
            allow_missing_columns: bool = False,
    ):
        """Append a dataframe to the main one in the pipeline.

        Args:
            store_key (str | None):
                Dataframe name in Nebula storage.
            allow_missing_columns (bool):
                When this parameter is True, the set of column names in the
                dataframe to append and in the main one can differ; missing
                columns will be filled with null.
                Further, the missing columns of this DataFrame will be
                added at the end of the union result schema.
                This parameter was introduced in spark 3.1.0.
                If it is set to True with a previous version, it throws an error.
                Defaults to False.
        """
        super().__init__()
        self._store_key: str | None = store_key
        self._allow_missing: bool = allow_missing_columns

    def _transform_nw(self, df):
        df_union = ns.get(self._store_key)

        if not isinstance(df_union, (nw.DataFrame, nw.LazyFrame)):
            df_union = nw.from_native(df_union)

        cols_main = set(df.columns)
        cols_union = set(df_union.columns)
        diff = cols_main.symmetric_difference(cols_union)

        if not diff:
            return nw.concat([df, df_union], how="vertical")

        # If differences exist but not allowed, raise error
        if not self._allow_missing:
            missing_in_main = cols_union - cols_main
            missing_in_union = cols_main - cols_union
            msg = "Column mismatch between dataframes. "
            if missing_in_main:
                msg += f"Missing in main df: {sorted(missing_in_main)}. "
            if missing_in_union:
                msg += f"Missing in union df: {sorted(missing_in_union)}."
            raise ValueError(msg)

        df_native = nw.to_native(df)
        if get_dataframe_type(df) == "pandas":
            import pandas as pd
            # Let pandas allow the missing columns in the best manner
            if isinstance(df_union, (nw.LazyFrame, nw.DataFrame)):
                df_union = nw.to_native(df_union)
            ret = pd.concat([df_native, df_union], axis=0)
            return nw.from_native(ret)

        # Add missing columns with nulls
        missing_in_main = cols_union - cols_main
        missing_in_union = cols_main - cols_union

        if missing_in_main:
            union_schema = df_union.schema
            df = df.with_columns(*[
                nw.lit(None).cast(union_schema[col]).alias(col)
                for col in sorted(missing_in_main)
            ])

        if missing_in_union:
            df_schema = df.schema
            df_union = df_union.with_columns(*[
                nw.lit(None).cast(df_schema[col]).alias(col)
                for col in sorted(missing_in_union)
            ])

        # Align column order: main columns first, then union-only columns
        final_order = list(df.columns)
        df = df.select(final_order)
        df_union = df_union.select(final_order)
        return nw.concat([df, df_union], how="vertical")


class DropNulls(Transformer):
    def __init__(
            self,
            *,
            how: str = "any",
            thresh: int | None = None,
            drop_na: bool = False,
            columns: str | list[str] | None = None,
            regex: str | None = None,
            glob: str | None = None,
            startswith: str | Iterable[str] | None = None,
            endswith: str | Iterable[str] | None = None,
    ):
        """Drop rows with null values.

        Input parameters are eventually used to select a subset of the columns.

        Args:
            how (str):
                "any" or "all". If "any", drop a row if it contains any nulls.
                If "all", drop a row only if all its values are null.
                Defaults to any.
            thresh (int | None):
                Require that many non-NA values. Cannot be combined with how.
                Used with Pandas and Spark. Ignored with Polars.
                Defaults to None.
            drop_na (bool):
                Used only with Polars, if True, treat NaN as nulls and drop them.
                Ignored with Pandas and Spark. Default to False.
            columns (str | list(str) | None):
                List of columns to consider. Defaults to None.
            regex (str | None):
                Take the columns to consider by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Take the columns to consider by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns to consider whose names start with the
                provided string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns to consider whose names end with the
                provided string(s). Defaults to None.
        """
        assert_allowed(how, {"any", "all"}, "how")
        super().__init__()
        self._how: str = how
        self._thresh: int | None = thresh
        self._drop_na: bool = drop_na
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )

    def _transform_pandas(self, df):
        subset: list[str] = self._get_selected_columns(df)
        if subset and set(subset) != set(list(df.columns)):
            return df.dropna(self._how, subset=subset, thresh=self._thresh)
        return df.dropna(self._how, thresh=self._thresh)

    def _transform_polars(self, df):
        import polars as pl

        subset: list[str] = self._get_selected_columns(df)
        cols = pl.col(*subset) if subset else pl.all()

        meth = pl.all_horizontal if self._how == "all" else pl.any_horizontal

        cond = meth(cols.is_null())

        # Add NaN check only for numeric columns
        if self._drop_na:
            # Get numeric columns from selection
            if subset:
                numeric_cols = [c for c in subset if df[c].dtype in pl.NUMERIC_DTYPES]
            else:
                numeric_cols = [c for c in df.columns if df[c].dtype in pl.NUMERIC_DTYPES]

            if numeric_cols:
                cond |= meth(pl.col(*numeric_cols).is_nan())

        return df.filter(~cond)

    def _transform_spark(self, df):
        subset: list[str] = self._get_selected_columns(df)
        if subset and set(subset) != set(list(df.columns)):
            return df.dropna(self._how, subset=subset, thresh=self._thresh)
        return df.dropna(self._how, thresh=self._thresh)


class Filter(Transformer):

    def __init__(
            self,
            *,
            input_col: str,
            perform: str,
            operator: str,
            value=None,
            compare_col: str | None = None,
    ):
        """Row filtering using Narwhals conditions.

        Filter rows based on a condition applied to a single column. Supports both
        value-based and column-to-column comparisons, with flexible 'keep' or 'remove'
        semantics.

        Args:
            input_col: Name of the column to filter on.
            operator: Comparison operator. Supported operators:

                **Standard comparisons** (work with value or compare_col):
                    - "eq": Equal to
                    - "ne": Not equal to
                    - "lt": Less than
                    - "le": Less than or equal to
                    - "gt": Greater than
                    - "ge": Greater than or equal to

                **Null/NaN checks** (no value or compare_col needed):
                    - "is_null": Column value is null (None)
                    - "is_not_null": Column value is not null
                    - "is_nan": Column value is NaN (float NaN, distinct from null)
                    - "is_not_nan": Column value is not NaN

                **String operations** (require string value):
                    - "contains": Column contains substring
                    - "starts_with": Column starts with string
                    - "ends_with": Column ends with string

                **Set membership** (require iterable value):
                    - "is_in": Column value is in the provided list/set
                    - "is_not_in": Column value is not in the provided list/set

                **Range check** (requires 2-element list/tuple):
                    - "is_between": Column value is between [lower, upper] (inclusive)

            value: Value to compare against. Required for most operators except null/NaN
                checks. Cannot be used together with compare_col.

                Type requirements by operator:
                    - Standard comparisons: any comparable type
                    - String operations: str
                    - is_in/is_not_in: list, tuple, or set (cannot contain None)
                    - is_between: list or tuple of exactly 2 elements [lower, upper]

            compare_col: Name of another column to compare against. Allows column-to-column
                comparisons (e.g., sales > target). Cannot be used together with value.
                Not supported for string operations or is_between.

            perform: Whether to "keep" or "remove" rows matching the condition.
                - "keep" (default): Keep rows where condition is True, exclude others
                - "remove": Remove rows where condition is True, keep others

                **Important:** Cannot combine perform="remove" with negative operators
                (ne, is_not_in, is_not_null, is_not_nan) as this creates confusing double
                negation. Use perform="keep" with the opposite operator instead.

        Raises:
            ValueError: If invalid operator, incompatible parameters, or double negation
                is attempted (perform="remove" with ne/is_not_in/is_not_null/is_not_nan).
            TypeError: If value has wrong type for the operator.

        Notes:
            **Null Handling:**
            - Standard comparisons (eq, ne, lt, etc.) with null values return null,
              which is excluded by filter. Example: `age > 18` excludes null ages.
            - The "ne" operator may seem to match nulls, but `null != value` returns
              null (not True), so nulls are excluded. Use is_null explicitly if needed.
            - When using perform="remove", nulls are typically KEPT (since they don't
              match the removal condition). Example: removing "active" status keeps
              null statuses.

            **NaN vs Null:**
            - is_null checks for None/null values only
            - is_nan checks for float NaN values only
            - These are distinct: a column can have both null and NaN values

            **String Operations with Nulls (Pandas Limitation):**
            - String operations (contains, starts_with, ends_with) may fail in pandas
              when the column contains null values, due to NumPy object dtype limitations.
            - Error: "Cannot use ignore_nulls=False in all_horizontal..."
            - Solution: Filter out nulls first with is_not_null, then apply string operation.
            - This is a known pandas/NumPy limitation, not a bug in Nebula.

            **Performance:**
            - Filters are pushed down to the underlying backend (pandas/Polars/Spark)
            - For large datasets, consider using is_between instead of combining
              operators (e.g., `is_between: [0, 100]` vs `ge: 0` + `le: 100`)

        Examples:
        Basic filtering:
            >>> # Keep adults
            >>> Filter(input_col="age", operator="gt", value=18)

            >>> # Remove inactive users
            >>> Filter(input_col="status", perform="remove", operator="eq", value="inactive")

        Null handling:
            >>> # Keep only rows with non-null age
            >>> Filter(input_col="age", operator="is_not_null")

            >>> # Remove rows with null email (keep all others)
            >>> Filter(input_col="email", perform="remove", operator="is_null")

        NaN handling:
            >>> # Remove NaN scores (keep numeric and null scores)
            >>> Filter(input_col="score", perform="remove", operator="is_nan")

        String operations:
            >>> # Keep company emails
            >>> Filter(input_col="email", operator="contains", value="@company.com")

            >>> # With nulls present (two-step approach for pandas):
            >>> # Step 1: Filter(input_col="email", operator="is_not_null")
            >>> # Step 2: Filter(input_col="email", operator="starts_with", value="admin")

        Set membership:
            >>> # Keep active or pending users
            >>> Filter(input_col="status", operator="is_in", value=["active", "pending"])

            >>> # Remove archived or deleted (keeps nulls!)
            >>> Filter(
            ...     input_col="status",
            ...     perform="remove",
            ...     operator="is_in",
            ...     value=["archived", "deleted"]
            ... )

        Range checks:
            >>> # Keep scores between 0 and 100 (inclusive)
            >>> Filter(input_col="score", operator="is_between", value=[0, 100])

        Column comparisons:
            >>> # Keep rows where sales exceed target
            >>> Filter(input_col="sales", operator="gt", compare_col="target")

            >>> # Remove rows where actual equals expected
            >>> Filter(
            ...     input_col="actual",
            ...     perform="remove",
            ...     operator="eq",
            ...     compare_col="expected"
            ... )

        Avoiding double negation:
            >>> # WRONG - double negation is confusing and disallowed:
            >>> # Filter(input_col="status", perform="remove", operator="is_not_in", value=["active"])
            >>>
            >>> # CORRECT - use positive logic:
            >>> Filter(input_col="status", operator="is_in", value=["active"])

        See Also:
            - When: For creating new columns with conditional logic
            - DropNulls: For removing rows with any null values across multiple columns
            - get_condition: The underlying function that builds filter conditions
        """
        # Prevent confusing double negatives
        if perform == "remove" and operator in {
            "ne", "is_not_in", "is_not_null", "is_not_nan"
        }:
            raise ValueError(
                f"Cannot use perform='remove' with operator '{operator}'. "
                f"This creates double negation which is confusing. "
                f"Use perform='keep' with the opposite operator instead.\n"
                f"Example: Instead of perform='remove' + is_not_in, "
                f"use perform='keep' + is_in."
            )

        assert_allowed(perform, {"keep", "remove"}, "perform")
        validate_operation(operator, value, compare_col)

        super().__init__()
        self._input_col: str = input_col
        self._perform: str = perform
        self._operator: str = operator
        self._value = value
        self._compare_col: str | None = compare_col

    def _transform_nw(self, df):
        # Build the condition
        condition = get_condition(
            self._input_col,
            self._operator,
            value=self._value,
            compare_col=self._compare_col,
        )
        if self._perform == "remove":
            condition = ~null_cond_to_false(condition)
        return df.filter(condition)


class GroupBy(Transformer):
    _msg_err = (
        "'prefix' and 'suffix' are allowed only for single "
        "aggregation on multiple columns, like "
        "{'sum': ['col_1', 'col_2']}"
    )

    _ALLOWED_GROUPBY_AGG = {
        m for m in dir(nw.col())
        if m.islower() and not m.startswith('_')
    }

    def __init__(
            self,
            *,
            aggregations: dict[str, list[str]] | dict[str, str] | list[dict[str, str]],
            groupby_columns: str | list[str] | None = None,
            groupby_regex: str | None = None,
            groupby_glob: str | None = None,
            groupby_startswith: str | Iterable[str] | None = None,
            groupby_endswith: str | Iterable[str] | None = None,
            prefix: str = "",
            suffix: str = "",
    ):
        """Perform GroupBy aggregation operations.

        Supports two aggregation syntaxes:
        1. Single aggregation on multiple columns with prefix/suffix:
           {"sum": ["col_1", "col_2"]}
        2. Explicit list of aggregation dictionaries:
           [{"agg": "sum", "col": "dollars", "alias": "total"}]

        Examples:
            >>> # Simple sum with prefix
            >>> t = GroupBy(
            ...     groupby_columns=["user_id"],
            ...     aggregations={"sum": ["sales", "revenue"]},
            ...     prefix="total_"
            ... )

            >>> # Multiple aggregations with explicit aliases
            >>> t = GroupBy(
            ...     groupby_columns=["user_id", "date"],
            ...     aggregations=[
            ...         {"agg": "sum", "col": "sales", "alias": "total_sales"},
            ...         {"agg": "mean", "col": "price", "alias": "avg_price"},
            ...         {"agg": "count", "col": "transaction_id", "alias": "n_transactions"}
            ...     ]
            ... )

            >>> # Group by columns matching a pattern
            >>> t = GroupBy(
            ...     groupby_regex="^id_",
            ...     aggregations={"sum": ["amount"]},
            ...     suffix="_total"
            ... )

        Args:
            aggregations: Two possible formats:
                1) Single aggregation on multiple columns:
                   {"sum": ["col_1", "col_2"]}
                   Use with prefix/suffix to create column aliases.
                2) List of aggregation dictionaries:
                   [{"agg": "sum", "col": "dollars", "alias": "total"}]
                   Keys "agg" and "col" are mandatory, "alias" is optional.
            groupby_columns: Columns to group by. Defaults to None.
            groupby_regex: Regex pattern to select groupby columns. Defaults to None.
            groupby_glob: Glob pattern to select groupby columns. Defaults to None.
            groupby_startswith: Select columns starting with string(s). Defaults to None.
            groupby_endswith: Select columns ending with string(s). Defaults to None.
            prefix: Prefix for aggregated column names (single aggregation only).
                Defaults to "".
            suffix: Suffix for aggregated column names (single aggregation only).
                Defaults to "".

        Raises:
            ValueError: If aggregation format is invalid.
            TypeError: If column or alias types are incorrect.
            ValueError: If no groupby selection is provided.
            ValueError: If prefix/suffix used with multiple aggregations.
        """
        assert_only_one_non_none(
            groupby_columns=groupby_columns,
            groupby_regex=groupby_regex,
            groupby_glob=groupby_glob,
            groupby_startswith=groupby_startswith,
            groupby_endswith=groupby_endswith
        )
        super().__init__()

        # Handle single-op syntax: {"sum": ["col_1", "col_2"]}
        if isinstance(aggregations, list) and len(aggregations) == 1:
            aggregations = self._check_single_op(aggregations[0], prefix, suffix)
        elif isinstance(aggregations, dict):
            aggregations = self._check_single_op(aggregations, prefix, suffix)
        else:
            if prefix or suffix:
                raise ValueError(self._msg_err)

        self._aggregations: list[dict[str, str]] = self._get_sanitized_aggregations(aggregations)

        self._set_columns_selections(
            columns=groupby_columns,
            regex=groupby_regex,
            glob=groupby_glob,
            startswith=groupby_startswith,
            endswith=groupby_endswith,
        )

    def _get_sanitized_aggregations(
            self, aggregations: dict[str, str] | list[dict[str, str]]
    ) -> list[dict[str, str]]:
        if isinstance(aggregations, dict):
            aggregations = [aggregations]

        self._validate_aggregations(
            aggregations,
            required_keys={"agg", "col"},
            allowed_keys={"agg", "col", "alias"},
        )
        return aggregations

    def _validate_aggregations(
            self,
            aggregations: list[dict[str, str]],
            *,
            required_keys: set[str],
            allowed_keys: set[str],
    ) -> None:
        """Validate the list of aggregations for groupBy operations."""
        for d in aggregations:
            keys = set(d)

            if not keys.issuperset(required_keys):
                raise ValueError(f"Mandatory keys for aggregations: {required_keys}")

            if not keys.issubset(allowed_keys):
                raise ValueError(f"Allowed keys for aggregations: {allowed_keys}")

            # Type checks
            alias = d.get("alias")
            if alias and (not isinstance(alias, str)):
                raise TypeError('If provided, "alias" must be <str>')

            if not isinstance(d["col"], str):
                raise TypeError('"col" must be <str>')

            try:
                assert_allowed(d["agg"], self._ALLOWED_GROUPBY_AGG, "aggregation")
            except ValueError as e:
                # Enhance error message
                raise ValueError(
                    f"{e}\nAvailable aggregations: {sorted(self._ALLOWED_GROUPBY_AGG)}"
                )

    def _check_single_op(
            self, o: dict, prefix: str, suffix: str
    ) -> dict[str, str] | list[dict[str, str]]:
        """Check if this is single-operation syntax and expand it."""
        values = list(o.values())
        n = len(values)

        if n == 1:
            v = values[0]
            if isinstance(v, list):
                # Single operation on multiple columns: {"sum": ["col_1", "col_2"]}
                op = list(o.keys())[0]
                ret = []
                for col_name in v:
                    d = {"col": col_name, "agg": op}
                    alias = f"{prefix}{col_name}{suffix}"
                    d["alias"] = alias
                    ret.append(d)
                return ret

        # Not single-op syntax
        if prefix or suffix:
            raise ValueError(self._msg_err)
        return o

    def _transform_nw(self, df):
        """Transform using Narwhals for multi-backend support."""
        groupby_cols: list[str] = self._get_selected_columns(df)

        # Build aggregation expressions
        agg_exprs = []
        for agg_dict in self._aggregations:
            col_name = agg_dict["col"]
            agg_func = agg_dict["agg"]
            alias = agg_dict.get("alias")

            # Get the aggregation method
            col_expr = nw.col(col_name)
            agg_expr = getattr(col_expr, agg_func)()

            # Apply alias if provided
            if alias:
                agg_expr = agg_expr.alias(alias)

            agg_exprs.append(agg_expr)

        # Perform groupby and aggregation
        return df.group_by(groupby_cols).agg(agg_exprs)


class Join(Transformer):
    def __init__(
            self,
            *,
            store_key: str,
            how: str,
            on: list[str] | str | None = None,
            left_on: str | list[str] | None = None,
            right_on: str | list[str] | None = None,
            suffix: str = "_right"
    ):
        """Joins with another DataFrame, using the given join expression.

        The right dataframe is retrieved from the nebula storage.

        Args:
            store_key (str):
                Nebula storage key to retrieve the right table of the join.
            how (str):
                Must be one of: 'inner', 'left', 'full', 'cross', 'semi',
                'anti', 'right', 'right_semi', 'right_anti'.
                Note: 'right', 'right_semi', and 'right_anti' are implemented
                by swapping the dataframes and using 'left', 'semi', 'anti'.
            on (list(str), str):
                A string for the join column name, or a list of column names.
                The name of the join column(s) must exist on both sides.
            left_on	(str | list[str] | None):
            	Join column of the left DataFrame.
            right_on (str | list[str] | None):
            	Join column of the right DataFrame.
            suffix (str):
                Suffix to append to columns with a duplicate name.
                Defaults to "right".
        """
        allowed_how = {
            "inner",
            "cross",
            "full",
            "left",
            "semi",
            "anti",
            # not narwhals
            "right",
            "rightsemi",
            "right_semi",
            "rightanti",
            "right_anti",
        }
        assert_allowed(how, allowed_how, "how")

        super().__init__()
        self._table: str = store_key
        self._how: str = how
        self._on = ensure_flat_list(on) if on else None
        self._left_on = ensure_flat_list(left_on) if left_on else None
        self._right_on = ensure_flat_list(right_on) if right_on else None
        self._suffix: str = suffix

    def _transform_nw(self, df):
        df_to_join = ns.get(self._table)

        if not isinstance(df_to_join, nw.DataFrame):
            df_to_join = nw.from_native(df_to_join)

        # Map right-side joins to left-side by swapping dataframes
        swap_map = {
            "right": "left",
            "rightsemi": "semi",
            "right_semi": "semi",
            "rightanti": "anti",
            "right_anti": "anti",
        }

        if self._how in swap_map:
            left, right = df_to_join, df
            how = swap_map[self._how]

            if self._left_on and self._right_on:
                left_on, right_on = self._right_on, self._left_on
            else:
                left_on, right_on = self._left_on, self._right_on

            on = self._on
        else:
            left, right = df, df_to_join
            how = self._how
            left_on, right_on = self._left_on, self._right_on
            on = self._on

        join_kwargs = {"how": how, "suffix": self._suffix}

        if on:
            join_kwargs["on"] = on
        elif left_on and right_on:
            join_kwargs["left_on"] = left_on
            join_kwargs["right_on"] = right_on

        return left.join(right, **join_kwargs)


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


class Pivot(Transformer):
    def __init__(
            self,
            *,
            pivot_col: str,
            id_cols: str | list[str] | None = None,
            id_regex: str | None = None,
            id_glob: str | None = None,
            id_startswith: str | Iterable[str] | None = None,
            id_endswith: str | Iterable[str] | None = None,
            aggregate_function: Literal[
                "min", "max", "first", "last", "sum", "mean", "median", "len"
            ] = "first",
            values_cols: str | list[str] | None = None,
            values_regex: str | None = None,
            values_glob: str | None = None,
            values_startswith: str | Iterable[str] | None = None,
            values_endswith: str | Iterable[str] | None = None,
            separator: str = "_",
    ):
        """Transform DataFrame from long to wide format (pivot operation).

        Args:
            pivot_col (str):
                Column whose unique values become new column names.
            id_cols (str | list(str) | None):
                Columns to use as identifiers (preserved in output).
            id_regex (str | None):
                Regex pattern to select id columns.
            id_glob (str | None):
                Glob pattern to select id columns.
            id_startswith (str | Iterable(str) | None):
                Select id columns starting with string(s).
            id_endswith (str | Iterable(str) | None):
                Select id columns ending with string(s).
            aggregate_function (str):
                Function to aggregate values when multiple rows have the same
                pivot/id combination. Options: "min", "max", "first", "last",
                "sum", "mean", "median", "len". Defaults to "first".
            values_cols (str | list(str) | None):
                Columns containing values to pivot (optional).
            values_regex (str | None):
                Regex pattern to select value columns (optional).
            values_glob (str | None):
                Glob pattern to select value columns (optional).
            values_startswith (str | Iterable(str) | None):
                Select value columns starting with string(s).
            values_endswith (str | Iterable(str) | None):
                Select value columns ending with string(s).
            separator (str):
                String to separate value column name from pivot value
                in new column names. Defaults to "_".

        Note:
            - At least one id column selector must be provided
            - If no values selectors are specified, all columns except
              pivot_col and id_cols will be pivoted
            - New column names format: "{value_col}{separator}{pivot_value}"

        Examples:
            >>> # Simple pivot with explicit columns
            >>> Pivot(
            ...     pivot_col="month",
            ...     id_cols=["product_id"],
            ...     values_cols="revenue",
            ...     aggregate_function="sum"
            ... )

            >>> # Pivot with regex selectors
            >>> Pivot(
            ...     pivot_col="category",
            ...     id_regex="^id_.*",
            ...     values_glob="metric_*",
            ...     aggregate_function="mean"
            ... )

            >>> # Pivot with startswith/endswith
            >>> Pivot(
            ...     pivot_col="quarter",
            ...     id_startswith="user",
            ...     values_endswith="_total",
            ...     aggregate_function="sum"
            ... )
        """
        super().__init__()
        assert_at_least_one_non_null(
            id_cols=id_cols,
            id_regex=id_regex,
            id_glob=id_glob,
            id_startswith=id_startswith,
            id_endswith=id_endswith
        )
        self._pivot_col = pivot_col
        self._aggregate_function = aggregate_function
        self._separator = separator

        # Store id column selectors
        self._id_cols = id_cols
        self._id_regex = id_regex
        self._id_glob = id_glob
        self._id_startswith = id_startswith
        self._id_endswith = id_endswith

        # Store value column selectors
        self._values_cols = values_cols
        self._values_regex = values_regex
        self._values_glob = values_glob
        self._values_startswith = values_startswith
        self._values_endswith = values_endswith

    def _transform_nw(self, df):
        # Select id columns
        self._set_columns_selections(
            columns=self._id_cols,
            regex=self._id_regex,
            glob=self._id_glob,
            startswith=self._id_startswith,
            endswith=self._id_endswith,
        )
        id_cols: list[str] = self._get_selected_columns(df)

        # Select value columns if any selector is specified
        values_cols: list[str] | None = None
        if any([
            self._values_cols,
            self._values_regex,
            self._values_glob,
            self._values_startswith,
            self._values_endswith,
        ]):
            self._set_columns_selections(
                columns=self._values_cols,
                regex=self._values_regex,
                glob=self._values_glob,
                startswith=self._values_startswith,
                endswith=self._values_endswith,
            )
            values_cols = self._get_selected_columns(df)

        # Narwhals pivot
        return df.pivot(
            on=self._pivot_col,
            index=id_cols if id_cols else None,
            values=values_cols,
            aggregate_function=self._aggregate_function,
            separator=self._separator,
        )


class Unpivot(Transformer):
    def __init__(
            self,
            *,
            id_cols: str | list[str] | None = None,
            id_regex: str | None = None,
            melt_cols: str | list[str] | None = None,
            melt_regex: str | None = None,
            variable_col: str,
            value_col: str,
    ):
        """Unpivot DataFrame from wide to long format.

        Args:
            id_cols: Columns to keep as identifiers
            id_regex: Regex pattern to select id columns
            melt_cols: Columns to unpivot into rows
            melt_regex: Regex pattern to select melt columns
            variable_col: Name for the new variable column
            value_col: Name for the new value column
        """
        assert_at_least_one_non_null(
            melt_cols=melt_cols,
            melt_regex=melt_regex
        )
        super().__init__()

        self._id_cols = id_cols
        self._id_regex = id_regex
        self._melt_cols = melt_cols
        self._melt_regex = melt_regex
        self._variable_col = variable_col
        self._value_col = value_col

    def _transform_nw(self, df):
        # Select id columns (to KEEP)
        self._set_columns_selections(
            columns=self._id_cols,
            regex=self._id_regex,
        )
        id_cols: list[str] = self._get_selected_columns(df)

        # Select melt columns (to UNPIVOT)
        self._set_columns_selections(
            columns=self._melt_cols,
            regex=self._melt_regex,
        )
        melt_cols: list[str] = self._get_selected_columns(df)

        # Narwhals unpivot (confusing parameter names!)
        # on = columns to melt
        # index = columns to keep as identifiers
        return df.unpivot(
            on=melt_cols,  # These get melted
            index=id_cols if id_cols else None,  # These stay as-is
            variable_name=self._variable_col,
            value_name=self._value_col,
        )


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
