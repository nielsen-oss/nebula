"""Row filtering operations."""

from typing import Iterable

from nlsn.nebula.auxiliaries import assert_allowed
from nlsn.nebula.base import Transformer
from nlsn.nebula.nw_util import validate_operation, get_condition, null_cond_to_false

__all__ = ["DropNulls", "Filter"]


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
        kws = {}
        # pandas cannot accept both 'how' and 'thresh'
        if self._thresh is not None:
            kws["thresh"] = self._thresh
        else:
            kws["how"] = self._how

        subset: list[str] = self._get_selected_columns(df)
        if subset and set(subset) != set(list(df.columns)):
            kws["subset"] = subset

        return df.dropna(**kws)

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
