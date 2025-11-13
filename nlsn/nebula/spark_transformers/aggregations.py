"""GroupBy and Window Operations."""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from pyspark.sql import Window
from pyspark.sql import functions as F

from nlsn.nebula.auxiliaries import (
    assert_allowed,
    assert_at_most_one_args,
    assert_only_one_non_none,
    ensure_flat_list,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.spark_transformers._constants import (
    ALLOWED_GROUPBY_AGG,
    ALLOWED_WINDOW_AGG,
)

__all__ = [
    "AggregateOverWindow",
    "GroupBy",
    "LagOverWindow",
    "Pivot",
]


def _expand_ascending_windowing_cols(ascending, order_cols) -> List[str]:
    """Expand ascending to fill the missing orders."""
    n_ascending = len(ascending)
    n_order_cols = len(order_cols)

    if n_ascending == 1:
        return ascending * n_order_cols

    if n_order_cols != n_ascending:
        msg = f"Length of order columns: {n_order_cols} does not "
        msg += f"match length of sort criteria: {n_ascending}."
        raise AssertionError(msg)
    return ascending


_DOCSTRING_ARGS_WINDOW = """
            ascending (bool | list(bool)):
                It can be a <bool>, in this case the order will be same for
                all the column. Or it can be a list of <bool>, specifying
                which order to follow for each column. Defaults to True.
            rows_between (((str | int), (str | int)) | None):
                Creates a WindowSpec with the frame boundaries defined,
                from start (inclusive) to end (inclusive).
                Both start and end are relative positions from the current row.
                For example, “0” means “current row”, while “-1” means the row
                before the current row, and “5” means the fifth row after the
                current row.
                A row based boundary is based on the position of the row within
                the partition. An offset indicates the number of rows above
                or below the current row, the frame for the current row starts
                or ends. For instance, given a row based sliding frame with a
                lower bound offset of -1 and an upper bound offset of +2.
                The frame for row with index 5 would range from index 4 to
                index 7.
                Defaults to None.
            range_between (((str | int), (str | int)) | None):
                Creates a WindowSpec with the frame boundaries defined, from
                start (inclusive) to end (inclusive).
                Both start and end are relative from the current row. For
                example, “0” means “current row”, while “-1” means one off
                before the current row, and “5” means the five off after
                the current row.
                A range-based boundary is based on the actual value of the
                ORDER BY expression(s). An offset is used to alter the value
                of the ORDER BY expression, for instance if the current
                ORDER BY expression has a value of 10 and the lower bound
                offset is -3, the resulting lower bound for the current row
                will be 10 - 3 = 7. This however puts a number of constraints
                on the ORDER BY expressions: there can be only one expression
                and this expression must have a numerical data type. An
                exception can be made when the offset is unbounded, because
                no value modification is needed, in this case multiple and
                non-numeric ORDER BY expression are allowed.
                If provided, the parameter 'order_cols' must be passed as well.
                Defaults to None.

        Notes for 'rows_between' and 'range_between':
            To use the very first boundary for left boundary, the user can
            provide the string "start" instead of an integer. "start"
            represents the 'Window.unboundedPreceding' default value.
            To use the very last boundary for the right boundary, the user
            can provide the string "end" instead of an integer.
            "end" represents the 'Window.unboundedFollowing' default value.
"""


def validate_aggregations(
        o: List[Dict[str, str]],
        allowed_agg: Set[str],
        *,
        exact_keys: Optional[Set[str]] = None,
        required_keys: Optional[Set[str]] = None,
        allowed_keys: Optional[Set[str]] = None,
) -> None:
    """Validate the list of aggregations for groupBy and window functions."""
    for d in o:
        keys = set(d)
        if exact_keys and (keys != exact_keys):
            raise ValueError(f"Requested keys for aggregations: {exact_keys}")

        if allowed_keys and not keys.issubset(allowed_keys):
            raise ValueError(f"Allowed keys for aggregations: {allowed_keys}")

        if required_keys and not keys.issuperset(required_keys):
            raise ValueError(f"Mandatory keys for aggregations: {required_keys}")

        alias = d.get("alias")
        if alias and (not isinstance(alias, str)):
            raise TypeError('If provided, "alias" must be <str>')
        if not isinstance(d["col"], str):
            raise TypeError('"col" must be <str>')
        assert_allowed(d["agg"], allowed_agg, "aggregation")


def validate_window_frame_boundaries(start, end) -> Tuple[int, int]:
    """Validate the window frame boundaries."""
    if (start is None) or (end is None):
        raise ValueError("'start' and 'end' cannot be None")
    if isinstance(start, str):
        if start != "start":
            raise ValueError("if 'start' is <str> must be 'start'")
        start = Window.unboundedPreceding
    if isinstance(end, str):
        if end != "end":
            raise ValueError("if 'end' is <str> must be 'end'")
        end = Window.unboundedFollowing
    return start, end


def _make_aggregations(aggregations: List[Dict[str, str]]) -> List[F.col]:
    """Creates a list of Spark SQL column expressions for aggregations.

    Args:
        aggregations (list(dict(str, str))):
            A list of dictionaries where each dictionary contains the
            aggregation function ('agg'), the column name ('col'),
            and an optional alias ('alias').

    Returns (list(pyspark.sql.function.column)):
        A list of Spark SQL column expressions with the specified aggregations
        and aliases.
    """
    list_agg: List[F.col] = []
    for el in aggregations:
        agg: F.col = getattr(F, el["agg"])(el["col"])
        alias: str = el.get("alias")
        if alias:
            agg = agg.alias(alias)
        list_agg.append(agg)
    return list_agg


def _get_sanitized_aggregations(
        aggregations: Union[Dict[str, str], List[Dict[str, str]]]
) -> List[Dict[str, str]]:
    if isinstance(aggregations, dict):
        aggregations = [aggregations]

    validate_aggregations(
        aggregations,
        ALLOWED_GROUPBY_AGG,
        required_keys={"agg", "col"},
        allowed_keys={"agg", "col", "alias"},
    )
    return aggregations


class _Window(Transformer):
    def __init__(
            self,
            *,
            partition_cols: Union[str, List[str], None],
            order_cols: Union[str, List[str], None],
            ascending: Union[bool, List[bool]],
            rows_between: Optional[Tuple[Union[str, int], Union[str, int]]],
            range_between: Optional[Tuple[Union[str, int], Union[str, int]]],
    ):
        if range_between and not order_cols:
            msg = "If 'range_between' is provided 'order_cols' must be set as well."
            raise AssertionError(msg)

        super().__init__()
        self._partition_cols: List[str] = ensure_flat_list(partition_cols)

        self._list_order_cols: List[str] = ensure_flat_list(order_cols)

        self._ascending: List[str] = _expand_ascending_windowing_cols(
            ensure_flat_list(ascending), self._list_order_cols
        )

        self._order_cols: List[F.col] = [
            F.col(i).asc() if j else F.col(i).desc()
            for i, j in zip(self._list_order_cols, self._ascending)
        ]

        self._rows_between: Optional[Tuple[int, int]] = None
        if rows_between:
            self._rows_between = validate_window_frame_boundaries(*rows_between)

        self._range_between: Optional[Tuple[int, int]] = None
        if range_between:
            self._range_between = validate_window_frame_boundaries(*range_between)

    @property
    def _get_window(self):
        window = Window
        if self._partition_cols:
            window = window.partitionBy(self._partition_cols)

        if self._order_cols:
            window = window.orderBy(self._order_cols)

        if self._rows_between:
            window = window.rowsBetween(*self._rows_between)

        if self._range_between:
            window = window.rangeBetween(*self._range_between)

        return window


class AggregateOverWindow(_Window):
    def __init__(
            self,
            *,
            partition_cols: Union[str, List[str], None] = None,
            aggregations: Union[List[Dict[str, str]], Dict[str, str]],
            order_cols: Union[str, List[str], None] = None,
            ascending: Union[bool, List[bool]] = True,
            rows_between: Optional[Tuple[Union[str, int], Union[str, int]]] = None,
            range_between: Optional[Tuple[Union[str, int], Union[str, int]]] = None,
    ):  # noqa: D208, D209
        """Aggregate over a window.

        It returns the original dataframe with attached new columns defined
        in aggregations.

        Args:
            partition_cols (str | list(str) | None):
                Columns to partition on. If set to None, the entire DataFrame is
                considered as a single partition. Defaults to None.
            aggregations (list(dict(str, str)) | dict(str, str)):
                A list of aggregation dictionaries to be applied.
                If a single aggregation is provided (equivalent to a list of
                length=1), it can be a flat dictionary.
                Each aggregation is defined with the following fields:
                'col': (the column to aggregate)
                'agg': (the aggregation operation)
                'alias': (the alias for the aggregated column)
                Eg:
                [
                    {"agg": "avg", "col": "dollars", "alias": "mean_dollars"},
                    {"agg": "sum", "col": "dollars", "alias": "tot_dollars"},
                ]
                "alias" is a necessary key, and to prevent unexpected behavior
                it cannot be a column used in neither 'partition_col' nor in
                'order_cols', even though it can a column used in
                'aggregations', but keep in mind that aggregations are computed
                sequentially, not in parallel.
            order_cols (str | list(str) | None):
                Columns to order the partition by. If provided, the partition
                will be ordered based on these columns. Defaults to None."""
        assert_at_most_one_args(rows_between, range_between)

        if isinstance(aggregations, dict):
            aggregations = [aggregations]

        validate_aggregations(
            aggregations, ALLOWED_WINDOW_AGG, exact_keys={"agg", "col", "alias"}
        )

        super().__init__(
            partition_cols=partition_cols,
            order_cols=order_cols,
            ascending=ascending,
            rows_between=rows_between,
            range_between=range_between,
        )

        self._aggregations: List[Dict[str, str]] = aggregations
        self._aliases: Set[str] = {i["alias"] for i in self._aggregations}
        self._check_alias_override(self._partition_cols, "partition_cols")
        self._check_alias_override(self._list_order_cols, "order_cols")

    def _check_alias_override(self, cols: List[str], name: str):
        ints = self._aliases.intersection(cols)
        if ints:
            raise AssertionError(f'Some aliased override "{name}": {ints}')

    def _transform(self, df):
        list_agg: List[Tuple[F.col, str]] = []
        for el in self._aggregations:
            agg: F.col = getattr(F, el["agg"])(el["col"])
            alias: str = el["alias"]
            list_agg.append((agg, alias))

        # Check duplicate names for final cols, in case we keep only aliases
        return_cols = [i for i in df.columns if i not in self._aliases]

        ints = self._aliases & set(df.columns)
        if ints:
            logger.warning(f"Overlapping column names: {ints} - keeping only the alias")

        win = self._get_window
        windowed_cols = [c.over(win).alias(alias) for c, alias in list_agg]

        return df.select(*return_cols, *windowed_cols)


class GroupBy(Transformer):
    _msg_err = (
        "'prefix' and 'suffix' are allowed only for single "
        "aggregation on multiple columns, like "
        "{'sum': ['col_1', 'col_2']}"
    )

    def __init__(
            self,
            *,
            aggregations: Union[Dict[str, List[str]], Dict[str, str], List[Dict[str, str]]],
            groupby_columns: Union[str, List[str], None] = None,
            groupby_regex: Optional[str] = None,
            groupby_glob: Optional[str] = None,
            groupby_startswith: [Union[str, Iterable[str]], None] = None,
            groupby_endswith: [Union[str, Iterable[str]], None] = None,
            prefix: str = "",
            suffix: str = "",
    ):
        """Performs a GroupBy operation.

        Args:
            aggregations (dict(str, list(str)), dict(str, str) | list(dict(str, str))):
                Two possible aggregation syntax are possible:
                1) A single aggregation on multiple columns, by providing
                a dictionary of only one key-value like:
                {"sum": ["col_1", "col_2"]}
                It will aggregate the two columns with a single "sum" operation.
                The user can provide the "prefix" and the "suffix" to create
                alias of the aggregated columns.
                2) A list of aggregation dictionaries to be applied.
                Each aggregation is defined with the following fields:
                'col' (the column to aggregate)
                'agg' (the aggregation operation)
                'alias' (the alias for the aggregated column)
                Eg:
                [
                    {"agg": "collect_list", "col": "time_bin"},
                    {"agg": "sum", "col": "dollars", "alias": "tot_dollars"},
                ]
                The keys "agg" and "col" are mandatory, whereas the key
                "alias" is optional.
                "prefix" and "suffix" are not allowed in this configuration.
            groupby_columns (str | list(str) | None):
                A list of the objective columns to groupby. Defaults to None.
            groupby_regex (str | None):
                Select the objective columns to groupby by using a regex pattern.
                Defaults to None.
            groupby_glob (str | None):
                Select the objective columns to groupby by using a bash-like pattern.
                Defaults to None.
            groupby_startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            groupby_endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
            prefix (str):
                Prefix to add to the aggregated column names when a single
                aggregation is called on multiple fields with an input like:
                {"sum": ["col_1", "col_2"]}.
                It raises a ValueError if provided for multiple aggregations.
                Defaults to "".
            suffix (str):
                Same as prefix.

        Raises:
            ValueError: if any aggregation is invalid.
            TypeError: if any column or alias in the aggregation is not a string type.
            ValueError: if no groupby selection is provided.
            ValueError: if 'prefix' or 'suffix' is provided for multiple aggregations.
        """
        assert_only_one_non_none(groupby_columns, groupby_regex, groupby_glob)
        super().__init__()

        self._aggregations: List[Dict[str, str]]

        if isinstance(aggregations, list) and len(aggregations) == 1:
            aggregations = self._check_single_op(aggregations[0], prefix, suffix)
        elif isinstance(aggregations, dict):
            aggregations = self._check_single_op(aggregations, prefix, suffix)
        else:
            if prefix or suffix:
                raise ValueError(self._msg_err)

        self._aggregations = _get_sanitized_aggregations(aggregations)
        self._set_columns_selections(
            columns=groupby_columns,
            regex=groupby_regex,
            glob=groupby_glob,
            startswith=groupby_startswith,
            endswith=groupby_endswith,
        )

    def _check_single_op(
            self, o: dict, prefix: str, suffix: str
    ) -> Union[Dict[str, str], List[Dict[str, str]]]:
        self._single_op = False
        values = list(o.values())
        n = len(values)
        if n == 1:
            v = values[0]
            if isinstance(v, list):
                # Eg: {"sum": ["col_1", "col_2"]}
                self._single_op = True
                op = list(o.keys())[0]
                ret = []
                for col_name in v:
                    d = {"col": col_name, "agg": op}
                    alias = f"{prefix}{col_name}{suffix}"
                    d["alias"] = alias
                    ret.append(d)

                return ret
        if prefix or suffix:
            raise ValueError(self._msg_err)
        return o

    def _transform(self, df):
        groupby_cols: List[str] = self._get_selected_columns(df)
        list_agg: List[F.col] = _make_aggregations(self._aggregations)
        return df.groupBy(groupby_cols).agg(*list_agg)


class LagOverWindow(_Window):
    def __init__(
            self,
            *,
            partition_cols: Union[str, List[str], None] = None,
            order_cols: Union[str, List[str], None] = None,
            lag_col: str,
            lag: int,
            output_col: str,
            ascending: Union[bool, List[bool]] = True,
            rows_between: Optional[Tuple[Union[str, int], Union[str, int]]] = None,
            range_between: Optional[Tuple[Union[str, int], Union[str, int]]] = None,
    ):  # noqa: D208, D209
        """Aggregate over a window.

        It returns the original dataframe with attached new columns defined
        in aggregations.

        Args:
            partition_cols (str | list(str) | None):
                Columns to partition on. If set to None, the entire DataFrame is
                considered as a single partition. Defaults to None.
            order_cols (str | list(str) | None):
                Columns to order the partition by. If provided, the partition
                will be ordered based on these columns. Defaults to None.
            lag_col (str):
                Column to be windowed by the lag defined in the 'lag' parameter.
            output_col (str):
                Name of the output column containing the windowed result.
        """
        assert_at_most_one_args(rows_between, range_between)

        super().__init__(
            partition_cols=partition_cols,
            order_cols=order_cols,
            ascending=ascending,
            rows_between=rows_between,
            range_between=range_between,
        )

        self._lag_col: str = lag_col
        self._output_col: str = output_col
        self._lag: int = lag

    def _transform(self, df):
        win = self._get_window
        return df.withColumn(
            self._output_col, F.lag(self._lag_col, self._lag).over(win)
        )


class Pivot(Transformer):
    def __init__(
            self,
            *,
            pivot_col: str,
            aggregations: Union[Dict[str, str], List[Dict[str, str]]],
            groupby_columns: Union[str, List[str], None] = None,
            distinct_values: Union[str, List[str], None] = None,
            groupby_regex: Optional[str] = None,
            groupby_glob: Optional[str] = None,
            groupby_startswith: Union[str, Iterable[str], None] = None,
            groupby_endswith: Union[str, Iterable[str], None] = None,
    ):
        """Pivots a column of the current DataFrame and perform the specified aggregation.

        There are two versions of the pivot function: one that requires the
        caller to specify the list of distinct values to pivot on, and one
        that does not. The latter is more concise but less efficient
        because Spark needs to first compute the list of distinct values
        internally.

        Args:
            pivot_col (str):
                Name of the column to pivot.
            aggregations (dict(str, str) | list(dict(str, str))):
                A dictionary list of aggregations to be applied.
                Each aggregation is defined with the following fields:
                'col' (the column to aggregate)
                'agg' (the aggregation operation)
                'alias' (the alias for the aggregated column)
                Eg:
                [
                    {"agg": "collect_list", "col": "time_bin"},
                    {"agg": "sum", "col": "dollars", "alias": "tot_dollars"},
                ]
                The keys "agg" and "col" are mandatory, whereas the key
                "alias" is optional.
            groupby_columns (str | list(str) | None):
                A list of the objective columns to groupby. Defaults to None.
            distinct_values (str | list(str) | None):
                Specify the list of distinct values to pivot on.
                If not provided, Spark needs to first compute the list of
                distinct values internally, slowing down performance a bit.
            groupby_regex (str | None):
                Select the objective columns to groupby by using a regex pattern.
                Defaults to None.
            groupby_glob (str | None):
                Select the objective columns to groupby by using a bash-like pattern.
                Defaults to None.
            groupby_startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            groupby_endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.

        Raises:
            ValueError: if any aggregation is invalid.
            TypeError: if any column or alias in the aggregation is not a string type.
            ValueError: if no groupby selection is provided.
        """
        assert_only_one_non_none(groupby_columns, groupby_regex, groupby_glob)

        super().__init__()
        self._pivot_col: str = pivot_col
        self._aggregations: List[Dict[str, str]]
        self._aggregations = _get_sanitized_aggregations(aggregations)
        self._distinct_values: List[str] = ensure_flat_list(distinct_values)
        self._set_columns_selections(
            columns=groupby_columns,
            regex=groupby_regex,
            glob=groupby_glob,
            startswith=groupby_startswith,
            endswith=groupby_endswith,
        )

    def _transform(self, df):
        group_by_cols: List[str] = self._get_selected_columns(df)
        df_grouped = df.groupby(group_by_cols)
        if self._distinct_values:
            df_pivoted = df_grouped.pivot(self._pivot_col, self._distinct_values)
        else:
            df_pivoted = df_grouped.pivot(self._pivot_col)

        list_agg: List[F.col] = _make_aggregations(self._aggregations)
        return df_pivoted.agg(*list_agg)


AggregateOverWindow.__init__.__doc__ = (
        AggregateOverWindow.__init__.__doc__ + _DOCSTRING_ARGS_WINDOW
)
LagOverWindow.__init__.__doc__ = LagOverWindow.__init__.__doc__ + _DOCSTRING_ARGS_WINDOW
