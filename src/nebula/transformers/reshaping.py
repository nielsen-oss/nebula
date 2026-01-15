"""DataFrame shape transformations."""

from typing import Iterable, Literal

import narwhals as nw

from nebula.auxiliaries import (
    assert_allowed,
    assert_at_least_one_non_null,
    assert_only_one_non_none,
)
from nebula.base import Transformer

__all__ = [
    "GroupBy",
    "Pivot",
    "Unpivot",
]


class GroupBy(Transformer):
    _msg_err = (
        "'prefix' and 'suffix' are allowed only for single "
        "aggregation on multiple columns, like "
        "{'sum': ['col_1', 'col_2']}"
    )

    _ALLOWED_GROUPBY_AGG: set[str] = {m for m in dir(nw.col()) if m.islower() and not m.startswith("_")}

    def __init__(  # noqa: PLR0913
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
            ...         {"agg": "count", "col": "transaction_id", "alias": "m"}
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
            groupby_regex: Regex pattern to select groupby columns.
                Defaults to None.
            groupby_glob: Glob pattern to select groupby columns.
                Defaults to None.
            groupby_startswith: Select columns starting with string(s).
                Defaults to None.
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
            groupby_endswith=groupby_endswith,
        )
        super().__init__()

        # Handle single-op syntax: {"sum": ["col_1", "col_2"]}
        if isinstance(aggregations, list) and len(aggregations) == 1:
            aggregations = self._check_single_op(aggregations[0], prefix, suffix)
        elif isinstance(aggregations, dict):
            aggregations = self._check_single_op(aggregations, prefix, suffix)
        elif prefix or suffix:
            raise ValueError(self._msg_err)

        self._aggregations: list[dict[str, str]] = self._get_sanitized_aggregations(aggregations)

        self._set_columns_selections(
            columns=groupby_columns,
            regex=groupby_regex,
            glob=groupby_glob,
            startswith=groupby_startswith,
            endswith=groupby_endswith,
        )

    def _get_sanitized_aggregations(self, aggregations: dict[str, str] | list[dict[str, str]]) -> list[dict[str, str]]:
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

            if not keys.issuperset(required_keys):  # pragma: no cover
                raise ValueError(f"Mandatory keys for aggregations: {required_keys}")

            if not keys.issubset(allowed_keys):  # pragma: no cover
                raise ValueError(f"Allowed keys for aggregations: {allowed_keys}")

            # Type checks
            alias = d.get("alias")
            if alias and (not isinstance(alias, str)):  # pragma: no cover
                raise TypeError('If provided, "alias" must be <str>')

            if not isinstance(d["col"], str):  # pragma: no cover
                raise TypeError('"col" must be <str>')

            try:
                assert_allowed(d["agg"], self._ALLOWED_GROUPBY_AGG, "aggregation")
            except ValueError as e:
                # Enhance error message
                raise ValueError(f"{e}\nAvailable aggregations: {sorted(self._ALLOWED_GROUPBY_AGG)}")

    def _check_single_op(self, o: dict, prefix: str, suffix: str) -> dict[str, str] | list[dict[str, str]]:
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

        agg_exprs = []
        for agg_dict in self._aggregations:
            col_name = agg_dict["col"]
            agg_func = agg_dict["agg"]
            alias = agg_dict.get("alias")

            # Get the aggregation method
            col_expr = nw.col(col_name)
            agg_expr = getattr(col_expr, agg_func)()

            if alias:
                agg_expr = agg_expr.alias(alias)

            agg_exprs.append(agg_expr)

        return df.group_by(groupby_cols).agg(agg_exprs)


class Pivot(Transformer):
    def __init__(  # noqa: PLR0913
        self,
        *,
        pivot_col: str,
        id_cols: str | list[str] | None = None,
        id_regex: str | None = None,
        id_glob: str | None = None,
        id_startswith: str | Iterable[str] | None = None,
        id_endswith: str | Iterable[str] | None = None,
        aggregate_function: Literal["min", "max", "first", "last", "sum", "mean", "median", "len"] = "first",
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
        assert_at_least_one_non_null(
            id_cols=id_cols,
            id_regex=id_regex,
            id_glob=id_glob,
            id_startswith=id_startswith,
            id_endswith=id_endswith,
        )
        super().__init__()
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
        if any(
            [
                self._values_cols,
                self._values_regex,
                self._values_glob,
                self._values_startswith,
                self._values_endswith,
            ]
        ):
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
        assert_at_least_one_non_null(melt_cols=melt_cols, melt_regex=melt_regex)
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

        # on = columns to melt
        # index = columns to keep as identifiers
        return df.unpivot(
            on=melt_cols,  # These get melted
            index=id_cols if id_cols else None,  # These stay as-is
            variable_name=self._variable_col,
            value_name=self._value_col,
        )
