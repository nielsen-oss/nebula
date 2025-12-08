"""GroupBy and Window Operations."""

from typing import Iterable

from pyspark.sql import functions as F

from nlsn.nebula.auxiliaries import (
    assert_only_one_non_none,
    validate_aggregations,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_transformers._constants import (
    ALLOWED_GROUPBY_AGG,
)

__all__ = [
    "GroupBy",
]


def _make_aggregations(aggregations: list[dict[str, str]]) -> list[F.col]:
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
    list_agg: list[F.col] = []
    for el in aggregations:
        agg: F.col = getattr(F, el["agg"])(el["col"])
        alias: str = el.get("alias")
        if alias:
            agg = agg.alias(alias)
        list_agg.append(agg)
    return list_agg


def _get_sanitized_aggregations(
        aggregations: dict[str, str] | list[dict[str, str]]
) -> list[dict[str, str]]:
    if isinstance(aggregations, dict):
        aggregations = [aggregations]

    validate_aggregations(
        aggregations,
        ALLOWED_GROUPBY_AGG,
        required_keys={"agg", "col"},
        allowed_keys={"agg", "col", "alias"},
    )
    return aggregations


class GroupBy(Transformer):
    _msg_err = (
        "'prefix' and 'suffix' are allowed only for single "
        "aggregation on multiple columns, like "
        "{'sum': ['col_1', 'col_2']}"
    )

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

        self._aggregations: list[dict[str, str]]

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
    ) -> dict[str, str] | list[dict[str, str]]:
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
        groupby_cols: list[str] = self._get_selected_columns(df)
        list_agg: list[F.col] = _make_aggregations(self._aggregations)
        return df.groupBy(groupby_cols).agg(*list_agg)
