"""General Purposes Transformers."""

import operator as py_operator
from functools import reduce
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Union

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DataType, MapType

from nlsn.nebula.auxiliaries import (
    assert_allowed,
    assert_at_least_one_non_null,
    assert_at_most_one_args,
    ensure_flat_list,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_util import (
    ALLOWED_SPARK_HASH,
    assert_col_type,
    ensure_spark_condition,
    get_column_data_type_name,
    get_spark_condition,
    get_spark_session,
    hash_dataframe,
    null_cond_to_false,
)
from nlsn.nebula.storage import nebula_storage as ns

__all__ = [
    "BooleanMarker",
    "Coalesce",
    "ColumnMethod",
    "ConcatColumns",
    "Explode",
    "FillNa",
    "HashDataFrame",
    "IterableSize",
    "Join",
    "LogicalOperator",
    "MapWithFallback",
    "Melt",
    "MonotonicallyIncreasingIndexColumn",
    "MultipleBooleanMarker",
    "MultipleLiterals",
    "ReplaceWithMap",
    "RowWiseGreatestOrLeast",
    "SqlFunction",
    "ToDF",
    "UnionByName",
    "When",
]


def _assert_no_null_keys(d: dict) -> None:
    """Asserts that a dictionary does not contain None as a key.

    Args:
        d (dict): The dictionary to check.

    Raises:
        KeyError: If None is found as a key in the dictionary.
    """
    if None in d:
        raise KeyError("None as mapping key is not allowed")


def validate_args_kwargs(
    args: Optional[list] = None, kwargs: Optional[Dict[str, Any]] = None
) -> None:
    """Validate args and kwargs."""
    if (args is not None) and (not isinstance(args, (tuple, list))):
        raise TypeError("'args' must be a <list> or <tuple>")
    if kwargs is not None:
        if not isinstance(kwargs, dict):
            raise TypeError("'kwargs' must be a <dict>")
        if not all(isinstance(k, str) for k in kwargs):
            raise TypeError("All keys in 'kwargs' must be <str>")


class BooleanMarker(Transformer):
    def __init__(
        self,
        *,
        input_col: str,
        operator: str,
        value: Optional[Any] = None,
        comparison_column: Optional[str] = None,
        output_col: Optional[str] = None,
    ):
        """Mark True rows according to the given condition.

        Args:
            input_col (str):
                Input column.
            operator (str):
                Valid operators:
                - "eq":             equal
                - "ne":             not equal
                - "le":             less equal
                - "lt":             less than
                - "ge":             greater equal
                - "gt":             greater than
                - "isin":           iterable of valid values
                - "isnotin":           iterable of valid values
                - "array_contains": has at least one instance of <value> in a <ArrayType> column
                - "contains":       has at least one instance of <value> in a <StringType> column
                - "startswith":     The row value starts with <value> in a <StringType> column
                - "endswith":       The row value ends with <value> in a <StringType> column
                - "between":        is between 2 values, inclusive
                - "like":           matches a pattern with <_> or <%>
                - "rlike":          matches a regex pattern
                - "isNull"          *
                - "isNotNull"       *
                - "isNaN"           *
                - "isNotNaN"        *

                * Does not require the optional "value" argument

            output_col (str):
                If not provided, the replacement takes place in the input_col.
            value (any | None):
                Value used for the comparison.
            comparison_column (str | None):
                Name of column to be compared with `input_col`.

            Either `value` or `comparison_column` (not both) must be provided
            for python operators that require a comparison value.
        """
        ensure_spark_condition(operator, value=value, compare_col=comparison_column)
        super().__init__()
        self._input_col: str = input_col
        self._op: str = operator
        self._value: Optional[Any] = value
        self._compare_col: Optional[Any] = comparison_column
        self._output_col: str = input_col if output_col is None else output_col

    def _transform(self, df):
        cond: F.col = get_spark_condition(
            df,
            self._input_col,
            self._op,
            value=self._value,
            compare_col=self._compare_col,
        )
        when_clause = F.when(cond, F.lit(True)).otherwise(F.lit(False))
        return df.withColumn(self._output_col, when_clause)


class Coalesce(Transformer):
    def __init__(
        self,
        *,
        output_col: str,
        columns: Optional[Union[str, List[str]]] = None,
        drop_input_cols: bool = False,
        treat_nan_as_null: bool = False,
        treat_blank_string_as_null: bool = False,
    ):
        """Coalesce given input_cols in the output_col and drop input_cols if needed.

        Args:
            output_col (str):
                Result of coalesce.
            columns (str | list(str) | None):
                A list of columns to coalesce. Defaults to None.
            drop_input_cols (bool):
                Drop input_cols after coalesce. If output_col is one of
                input_cols, it will not be dropped.
            treat_nan_as_null (bool):
                Spark treats NaN as number in numeric columns, hence if NaN
                occurs before a valid number, it will result in the output
                column. Set treat_nan_as_null as True if you want to treat
                NaN as None.
            treat_blank_string_as_null (bool):
                Treat empty and blank strings as None and coalesce accordingly.
        """
        assert_at_most_one_args(treat_nan_as_null, treat_blank_string_as_null)

        super().__init__()
        self._set_columns_selections(columns=columns)
        self._output_col: str = output_col
        self._drop_input_cols: bool = drop_input_cols
        self._treat_nan_as_null: bool = treat_nan_as_null
        self._treat_blank_string_as_null: bool = treat_blank_string_as_null

    def _transform(self, df):
        selection: List[str] = self._get_selected_columns(df)
        if self._treat_nan_as_null:
            cols = [
                F.when(F.isnan(i), None).otherwise(F.col(i)).alias(i) for i in selection
            ]
        elif self._treat_blank_string_as_null:
            cols = [
                F.when(F.trim(i) == "", None).otherwise(F.col(i)).alias(i)
                for i in selection
            ]
        else:
            cols = selection

        df = df.withColumn(self._output_col, F.coalesce(*cols))

        if self._drop_input_cols:
            cols2drop = [i for i in selection if i != self._output_col]
            df = df.drop(*cols2drop)

        return df


class ColumnMethod(Transformer):
    def __init__(
        self,
        *,
        input_column: str,
        output_column: Optional[str] = None,
        method: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Call a pyspark.sql.Column method with the provided args/kwargs.

        Args:
            input_column (str):
                Name of the input column.
            output_column (str):
                Name of the column where the result of the function is stored.
                If not provided, the input column will be used.
                Defaults to None.
            method (str):
                Name of the pyspark.sql.Column method to call.
            args (list(any) | None):
                Positional arguments of pyspark.sql.Column method.
                Defaults to None.
            kwargs (dict(str, any) | None):
                Keyword arguments of pyspark.sql.Column method.
                Defaults to None.
        """
        super().__init__()
        self._input_col: str = input_column
        self._output_col: str = output_column if output_column else input_column
        self._meth: str = method
        self._args: list = args if args else []
        self._kwargs: Dict[str, Any] = kwargs if kwargs else {}

        # Attempt to retrieve any errors during initialization.
        # Use a try-except block because Spark may not be running at this
        # point, making it impossible to guarantee the availability of the
        # requested method.
        self._assert_col_meth(False)

    def _assert_col_meth(self, raise_err: bool):
        try:
            all_meths = dir(F.col(self._input_col))
        except AttributeError as e:  # pragma: no cover
            if raise_err:
                raise e
            return

        valid_meths = {
            i for i in all_meths if (not i.startswith("_")) and (not i[0].isupper())
        }
        if self._meth not in valid_meths:
            raise ValueError(f"'method' must be one of {sorted(valid_meths)}")

    def _transform(self, df):
        self._assert_col_meth(True)
        func = getattr(F.col(self._input_col), self._meth)(*self._args, **self._kwargs)
        return df.withColumn(self._output_col, func)


class ConcatColumns(Transformer):
    def __init__(
        self,
        *,
        cols_to_concat: Optional[List[str]] = None,
        new_col_name: Optional[str] = None,
        separator: Optional[str] = None,
        null_if_any_null: bool = True,
        drop_initial_cols: bool = False,
        concat_strategy: Optional[List[dict]] = None,
    ):
        """Concatenate multiple columns with a given separator in a new one.

        Two strategies can be used:
        - A basic one, by providing the arguments:
            - cols_to_concat
            - new_col_name
            - separator
            - drop_initial_cols
        - A more comprehensive one, that covers a wider range of features,
            by providing only the argument 'concat_strategy'.

        Only one set of parameters can be provided, not both.

        Args:
            cols_to_concat (list(str) | None):
                Columns to be concatenated.
            new_col_name (str | None):
                Name of the new concatenated columns.
            separator (str | None):
                If provider, separate the column value with it.
            null_if_any_null (bool | None):
                If set to True, the output value is None if any null values
                exist in that row. Otherwise, null values are simply skipped,
                and if all the values are None, an empty string "" is returned.
                Ignored if 'concat_strategy' is provided.
            drop_initial_cols (bool):
                If True, the original columns are dropped from the returned df.
                Ignored if 'concat_strategy' is provided.
            concat_strategy (list(dict) | None):
                Strategy dictionary to concat columns:
                Example -> [
                    {
                        'new_column_name': 'new_col',
                        'separator': '_',
                        'strategy': [{'column': 'hh_id'}, {'constant': 'WE'}, {'constant': 'WD'}]
                    },
                    {
                        'new_column_name': 'new_col_2',
                        'separator': '$',
                        'strategy': [{'column': 'm5_agecat'}, {'constant': 'm5'}, {'column': 'm6_age_gap'}]
                    }
                ]

                This configuration will create two extra columns:
                    - new_col that is the concatenation between:
                        column "hh_id", constant "WE" and constant "WD"
                        separated by "_".
                    - new_col_2 that is the concatenation between:
                        column "m5_agecat", constant "m5" and column "m6_age_gap"
                        separated by "$".

                strategy list should contain dictionary of len == 1, and the keys
                of those dictionaries called _dict in the implementation must be
                'column' or 'constant'.
                By using this functionality, input columns cannot be dropped
                by this transformer and null values are treated as empty
                strings "".
        """
        self._use_concat_strategy: bool

        if concat_strategy is not None:
            if not (
                (cols_to_concat is None)
                and (new_col_name is None)
                and (separator is None)
            ):
                msg = 'If "concat_strategy" is provided, '
                msg += "the other arguments must not be passed."
                raise AssertionError(msg)

            self._check_concat_strategy(concat_strategy)
            self._use_concat_strategy = True
        else:
            self._use_concat_strategy = False

        super().__init__()
        self._cols_to_concat: Optional[List[str]] = cols_to_concat
        self._new_col_name: Optional[str] = new_col_name
        self._separator: Optional[str] = separator
        self._null_if_any_null: bool = null_if_any_null
        self._drop_initial_cols: bool = drop_initial_cols
        self._concat_strategy: Optional[List[dict]] = concat_strategy

    def _check_concat_strategy(self, concat_strategy: List[dict]):
        if not isinstance(concat_strategy, (list, tuple)):
            msg = '"concat_strategy" must be <list> or <tuple>'
            raise TypeError(msg)

        mandatory_keys = {"new_column_name", "separator", "strategy"}

        nd_outer: dict
        nd_inner: dict
        for nd_outer in concat_strategy:
            outer_keys = set(nd_outer)
            if outer_keys != mandatory_keys:
                msg = 'Wrong keys in "concat_strategy". '
                msg += f"Provided {outer_keys}, required: {mandatory_keys}."
                raise KeyError(msg)

            if not isinstance(nd_outer["new_column_name"], str):
                raise ValueError('"new_column_name" must be <str>')
            if not isinstance(nd_outer["separator"], str):
                raise ValueError('"separator" must be <str>')

            for nd_inner in nd_outer["strategy"]:
                self._check_inner_strategy_dict(nd_inner)

    @staticmethod
    def _check_inner_strategy_dict(d: dict):
        # Check len(d) == 1 and keys in {"column", "constant"}
        if len(d) != 1:
            raise ValueError(f"Len dict must be 1, found {d}")

        if not set(d).issubset({"column", "constant"}):
            msg = "Strategy lists should contain only column / constant as key"
            raise KeyError(msg)

    def _create_strategy_condition(self, d) -> F.col:
        self._check_inner_strategy_dict(d)
        key: str = list(d.keys())[0]
        value: str = d[key]
        return F.col(value) if key == "column" else F.lit(value)

    def _transform(self, df):
        if self._use_concat_strategy:
            nd: dict
            for nd in self._concat_strategy:
                strat = [self._create_strategy_condition(i) for i in nd["strategy"]]
                concat = F.concat_ws(nd["separator"], *strat)
                df = df.withColumn(nd["new_column_name"], concat)
            return df

        if self._null_if_any_null:
            _when_statement_cols = [c + " IS NOT NULL" for c in self._cols_to_concat]
            _when_statement = " AND ".join(c for c in _when_statement_cols)

            cond = F.expr(_when_statement)
            concat = F.concat_ws(self._separator, *self._cols_to_concat)
            out_col = F.when(cond, concat).otherwise(F.lit(None))
        else:
            out_col = F.concat_ws(self._separator, *self._cols_to_concat)

        df = df.withColumn(self._new_col_name, out_col)

        if self._drop_initial_cols:
            df = df.drop(*self._cols_to_concat)
        return df


class Explode(Transformer):
    def __init__(
        self,
        *,
        input_col: str,
        output_cols: Optional[Union[List[str], str]] = None,
        outer: bool = True,
        drop_after: bool = False,
    ):
        """Explode an array column into multiple rows.

        Args:
            input_col (str):
                Column to explode.
            output_cols (str | None):
                Where to store the values.
                If the Column to explode is an <ArrayType>, 'output_cols'
                can be null and the exploded values inside the input column.
                Otherwise, if the Column to explode is a <MapType>,
                'output_cols' must be a 2-element <list> or <tuple> of string,
                representing the key and the value respectively.
            outer (bool):
                Whether to perform an outer-explode (null values are preserved).
                If the Column to explode is an <ArrayType>, it will preserve
                empty arrays and produce a null value as output.
                If the Column to explode is an <MapType>, it will preserve empty
                dictionaries and produce a null values as key and value output.
                Defaults to True.
            drop_after (bool):
                If to drop input_column after the F.explode.
        """
        if isinstance(output_cols, (list, tuple)):
            n = len(output_cols)
            msg = "If 'output_cols' is an iterable it must "
            msg += "be a 2-element <list> or <tuple> of string."
            if n != 2:
                raise AssertionError(msg)
            if not all(isinstance(i, str) for i in output_cols):
                raise AssertionError(msg)

        super().__init__()
        self._input_col: str = input_col
        self._output_cols: Union[List[str], str] = output_cols or input_col
        self._outer: bool = outer
        self._drop_after: bool = drop_after

    def _transform(self, df):
        explode_method = F.explode_outer if self._outer else F.explode

        input_type: DataType = df.select(self._input_col).schema[0].dataType

        if isinstance(input_type, ArrayType):
            if not isinstance(self._output_cols, str):  # pragma: no cover
                msg = "If the column to explode is <ArrayType> the 'output_col' "
                msg += "parameter must be a <str>."
                raise AssertionError(msg)
            func = explode_method(self._input_col)
            ret = df.withColumn(self._output_cols, func)

        elif isinstance(input_type, MapType):
            if not isinstance(self._output_cols, (list, tuple)):
                msg = "If the column to explode is <MapType> the 'output_cols' "
                msg += "parameter must be a 2 element <list>/<tuple> of <str>."
                raise AssertionError(msg)
            not_exploded = [i for i in df.columns if i not in self._output_cols]
            exploded = explode_method(self._input_col).alias(*self._output_cols)
            ret = df.select(*not_exploded, exploded)

        else:
            msg = "Input type not understood. Accepted <ArrayType> and <MapType>"
            raise AssertionError(msg)

        # Only if input col is different from output col
        if self._drop_after and self._input_col != self._output_cols:
            ret = ret.drop(self._input_col)

        return ret


class FillNa(Transformer):
    def __init__(
        self,
        *,
        value: Union[int, float, str, bool, Dict[str, Union[int, float, str, bool]]],
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
    ):
        """Replace null values.

        Args:
            value (int | float | str | bool | dict(str), (int | float | str | bool):
                Value to replace null values with.
                If the 'value' is a dict, it must be a mapping from column name
                (string) to replacement value.
                The replacement value must be an int, float, boolean, or string.
                In this case, all the other arguments: 'columns', 'regex', and
                'glob' must be null.
                Otherwise, if 'value' is a scalar, a subset can be specified
                through the remaining arguments.
                Columns specified in subset that do not have a matching data type are
                ignored. For example, if a value is a string, and subset contains a
                non-string column, then the non-string column is simply ignored.
            columns (str | list(str) | None):
                A list of columns to select. Defaults to None.
            regex (str | None):
                Take the columns to select by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Take the columns to select by using a bash-like pattern.
                Defaults to None.
        """
        super().__init__()
        self._flag_mapping: bool = isinstance(value, dict)

        if bool(columns) or bool(regex) or bool(glob):
            if self._flag_mapping:
                msg = "If 'value' is <dict>, all the other parameters must be null."
                raise AssertionError(msg)

        scalar = Union[int, float, str, bool]
        self._value: Union[scalar, Dict[str, scalar]] = value
        self._set_columns_selections(columns=columns, regex=regex, glob=glob)

    def _transform(self, df):
        if self._flag_mapping:
            return df.na.fill(self._value)
        subset = self._get_selected_columns(df)
        return df.na.fill(self._value, subset=subset)


class HashDataFrame(Transformer):
    def __init__(
        self,
        *,
        output_col: str,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
        hash_name: str = "md5",
        num_bits: int = 256,
    ):
        """Hash each dataframe row and store the result in a new column.

        If no input columns to hash the dataframe are selected, all fields
        are used. All the columns are sorted before being hashed to ensure
        a repeatable result.

        Valid 'hash_name' function:
        - "md5"
        - "crc32"
        - "sha1"
        - "sha2"
        - "xxhash64"

        Args:
            output_col (str):
                Name of the new column to store the hash values.
            columns (str | list(str) | None):
                A list of columns to select for hashing. Defaults to None.
            regex (str | None):
                Select the columns to select for hashing by using a regex
                pattern. Defaults to None.
            glob (str | None):
                Select the columns to select for hashing by using a
                bash-like pattern. Defaults to None.
            startswith (str | iterable(str) | None):
                Select the columns for hashing whose names start with the
                provided string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select the columns for hashing whose names end with the
                provided string(s). Defaults to None.
            hash_name (str):
                Hash function name, allowed values: "md5", "crc32", "sha1",
                "sha2", "xxhash64". Defaults to "md5".
            num_bits (int):
                Number of bits for the SHA-2 hash.
                Permitted values: 0, 224, 256, 384, 512,
                Ignored if hash_name is not "sha2". Defaults to 256.
        """
        assert_allowed(hash_name, ALLOWED_SPARK_HASH, "hash_name")
        super().__init__()
        self._output_col: str = output_col
        self._hash_name: str = hash_name
        self._num_bits: int = num_bits
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )

    def _transform(self, df):
        selection: List[str] = self._get_selected_columns(df)
        if selection:
            df_input = df.select(selection)
        else:
            df_input = df

        hashed_col = hash_dataframe(
            df_input,
            self._hash_name,
            num_bits=self._num_bits,
            return_func=True,
        )

        return df.withColumn(self._output_col, hashed_col)


class IterableSize(Transformer):
    def __init__(self, *, input_col: str, output_col: str):
        """Return the length of the array or map stored in a column.

        Alias (deprecated): ArraySize.

        Args:
            input_col (str):
                Name of the array column to compute size.
                If the value is null, the output is -1 by default.
            output_col (str):
                Name of the output column containing the size of the input column.
        """
        super().__init__()
        self._input_col: str = input_col
        self._output_col: str = output_col

    def _transform(self, df):
        return df.withColumn(self._output_col, F.size(self._input_col))


class Join(Transformer):
    def __init__(
        self,
        *,
        table: str,
        on: Union[List[str], str],
        how: str,
        broadcast: bool = False,
    ):
        """Joins with another DataFrame, using the given join expression.

        The right dataframe is retrieved from the nebula storage.

        Args:
            table (str):
                Nebula storage key to retrieve the right table of the join.
            on (list(str, str)):
                A string for the join column name, or a list of column names.
                The name of the join column(s) must exist on both sides.
            how (str):
                Must be one of: inner, cross, outer, full, fullouter,
                full_outer, left, leftouter, left_outer, right, rightouter,
                right_outer, semi, leftsemi, left_semi, anti, leftanti and
                left_anti.
            broadcast (bool):
                Broadcast the right dataframe. Defaults to False.
        """
        allowed_how = {
            "inner",
            "cross",
            "outer",
            "full",
            "fullouter",
            "full_outer",
            "left",
            "leftouter",
            "left_outer",
            "right",
            "rightouter",
            "right_outer",
            "semi",
            "leftsemi",
            "left_semi",
            "anti",
            "leftanti",
            "left_anti",
        }
        if how not in allowed_how:
            raise ValueError(f"'how' must be one of the following: {how}")

        super().__init__()
        self._table: str = table
        self._on: List[str] = ensure_flat_list(on)
        self._how: str = how
        self._broadcast: bool = broadcast

    def _transform(self, df):
        df_right = ns.get(self._table)
        if self._broadcast:
            df_right = F.broadcast(df_right)
        return df.join(df_right, on=self._on, how=self._how)


class LogicalOperator(Transformer):
    def __init__(
        self,
        *,
        operator: str,
        output_col: str,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        allow_excess_columns: bool = True,
    ):
        """Combine columns using logical operators AND or OR.

        !!!
        Note that all 'None' values are replaced with False before applying
        the logical operator to avoid ambiguity.
        !!!

        Args:
            operator (str):
                Logical operator, "AND" or "OR".
                Case-insensitive.
            output_col (str):
                Output column for the logical operation.
            columns (str | list(str) | None):
                A list of columns to use for the logical operator.
                Defaults to None.
            regex (str | None):
                Select the columns to use for the logical operator
                by using a regex pattern. Defaults to None.
            glob (str | None):
                Select the columns to use for the logical operator by
                using a bash-like pattern. Defaults to None.
            allow_excess_columns (bool):
                Whether to allow 'columns' argument to list columns that are
                not present in the dataframe. Default True.
                If 'columns' contains columns that are not present in the
                DataFrame and 'allow_excess_columns' is set to False, raise
                an AssertionError.

        Raises:
            AssertionError: If `operator` is not "AND" or "OR".
            AssertionError: If `allow_excess_columns` is False, and the column
            list contains columns that are not present in the DataFrame.
        """
        op = operator.strip().lower()
        assert_allowed(op, {"or", "and"}, "condition (case insensitive)")

        super().__init__()
        self._op = py_operator.and_ if op == "and" else py_operator.or_
        self._output_col: str = output_col
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            allow_excess_columns=allow_excess_columns,
        )

    def _transform(self, df):
        selection: List[str] = self._get_selected_columns(df)

        for c in selection:
            assert_col_type(df, c, "boolean")

        input_cols = [null_cond_to_false(F.col(i)) for i in selection]
        cond = reduce(self._op, input_cols)
        return df.withColumn(self._output_col, cond)


class MapWithFallback(Transformer):
    def __init__(
        self,
        *,
        input_col: str,
        mapping: dict,
        default: Any,
        output_col: Optional[str] = None,
    ):
        """Maps a column using a dictionary and sets all unmapped values to a default value.

        Ensures that input and output types are the same. Values not found in
        the "mapping" dictionary are replaced with the default value. None/null
        values are not allowed as keys in the mapping dictionary.

        Args:
            input_col (str):
                Input column.
            mapping (dict):
                Dictionary for replacing.
            default (Any):
                Fallback value for unmapped input values. It can be None
            output_col (str | None):
                If not provided, the replacement takes place in the input_col.
        """
        _assert_no_null_keys(mapping)
        super().__init__()
        self._input_col: str = input_col
        self._map: dict = mapping
        self._default: Any = default
        self._output_col: Optional[str] = output_col

    def _transform(self, df):
        output_col_name: str
        if self._output_col:
            output_col_name = self._output_col
            df = df.withColumn(output_col_name, F.col(self._input_col))
        else:
            output_col_name = self._input_col

        output_col = F.col(output_col_name)
        fallback = F.lit(self._default)

        mapping_expr = F.create_map(*[F.lit(x) for x in chain(*self._map.items())])

        if None in self._map.values():
            cond = output_col.isin([*self._map.keys()])
            result = F.when(cond, mapping_expr[output_col]).otherwise(fallback)
        else:  # Faster, but cannot use it if None in dictionary values
            result = F.coalesce(mapping_expr[output_col], fallback)

        return df.withColumn(output_col_name, result)


class Melt(Transformer):
    def __init__(
        self,
        *,
        id_cols: Optional[Union[str, List[str]]] = None,
        id_regex: Optional[str] = None,
        melt_cols: Optional[Union[str, List[str]]] = None,
        melt_regex: Optional[str] = None,
        variable_col: str,
        value_col: str,
    ):
        """Perform a melt operation, converting specified columns from wide to long format (Unpivot).

        Args:
            id_cols (str | list(str) | None): Set of identifier columns.
            id_regex (str | None): A regex that selects columns to be used as `id_cols`.
            melt_cols (str | list(str)): List of column names to unpivot.
            melt_regex (str | None): A regex that selects columns to be used as `value_cols`.
            variable_col (str): Name of the new column to store variable names after melting.
            value_col (str): Name of the new column to store values after melting.

        Note:
          1. If neither `id_cols` nor `id_regex` is provided, all non-melt columns will be discarded.

        Raises:
            AssertionError: if neither of `melt_cols` and `melt_regex` are provided.
        """
        assert_at_least_one_non_null(melt_cols, melt_regex)

        super().__init__()
        self._id_cols = id_cols
        self._id_regex = id_regex
        self._melt_cols = melt_cols
        self._melt_regex = melt_regex
        self._variable_col = variable_col
        self._value_col = value_col

    def _transform(self, df):
        self._set_columns_selections(
            columns=self._id_cols,
            regex=self._id_regex,
        )
        id_cols: List[str] = self._get_selected_columns(df)

        self._set_columns_selections(
            columns=self._melt_cols,
            regex=self._melt_regex,
        )
        melt_cols: List[str] = self._get_selected_columns(df)

        _vars_and_vals = F.array(
            *(
                F.struct(
                    F.lit(c).alias(self._variable_col), F.col(c).alias(self._value_col)
                )
                for c in melt_cols
            )
        )

        melted_df = df.withColumn("_vars_and_vals", F.explode(_vars_and_vals))

        cols = id_cols + [
            F.col("_vars_and_vals")[x].alias(x)
            for x in [self._variable_col, self._value_col]
        ]
        return melted_df.select(*cols)


class MonotonicallyIncreasingIndexColumn(Transformer):
    def __init__(
        self,
        *,
        output_col: str,
        sequential: bool = False,
        start_index: int = 0,
    ):
        """Add a monotonically increasing index column.

        Args:
            output_col (str):
                Output column that will contain the monotonically increasing index.
            sequential (bool):
                Indicates whether the index should be sequential. Default False.
                It is performed using a Window function without partitioning,
                thus is slow.
            start_index (int):
                The starting index value. Default is `0`.
                Only used if `sequential` is `True`.
        """
        super().__init__()
        self._output_col: str = output_col
        self._sequential: bool = sequential
        self._start_index: int = start_index

    def _transform(self, df):
        if self._sequential:
            start: int = self._start_index - 1
            win = Window.orderBy(F.monotonically_increasing_id())
            value = F.row_number().over(win) + start
            return df.withColumn(self._output_col, value)

        return df.withColumn(self._output_col, F.monotonically_increasing_id())


class MultipleBooleanMarker(Transformer):
    def __init__(
        self,
        *,
        output_col: str,
        conditions: List[Dict[str, Any]],
        logical_operators: Optional[List[str]] = None,
    ):
        """Create a boolean column based on multiple conditions.

        Args:
            output_col (str):
                Output column name.
            conditions (list(dict(str, any))):
                List of dictionaries containing conditions, eg:
                [
                    {
                        "column": "c1",
                        "operator": "ne", # Not equal
                        "value": 2,
                    },
                    {
                        "column": "c2",
                        "operator": "isNotNull",
                    },
                    {
                        "column": "c3",
                        "operator": "le", # Less or equal
                        "comparison_column": "c_x",
                    },
                ]
                See Filter - 'operator' argument for more detailed information.
            logical_operators (list(str)):
                List of logical operator to compound the 'conditions',
                therefore a list of "and" (or "&") and "or" (or "|"), like:
                ["and", "or", "|"].
                The number of 'logical_operators' should be one less than the
                number of 'conditions'.
                Note: The conditions are compound from left to right:

        Example:
            conditions = [
                {
                    "column": "c1",
                    "operator": "ne", # Not equal
                    "value": 2,
                },
                {
                    "column": "c2",
                    "operator": "isNotNull",
                },
                {
                    "column": "c3",
                    "operator": "le", # Less or equal
                    "comparison_column": "c_x",
                },
            ]
            logical_operators = ["or", "&"]

            1. Compare 'c1' and 'c2' using the "or" operator.
            2. Compare the previous result and 'c3' using the "and" operator.
            -> (("c1" OR "c2") AND "c3")
        """
        super().__init__()

        self._operators = {
            "&": py_operator.and_,
            "|": py_operator.or_,
            "and": py_operator.and_,
            "or": py_operator.or_,
        }

        # Sanity checks
        logical_operators = logical_operators or []
        n_cond = len(conditions)
        n_ops = len(logical_operators)
        if n_cond != (n_ops + 1):
            msg = "The number of ‘logical_operators’ should be one less "
            msg += "than the number of ‘conditions’."
            msg += f"Found {n_cond} 'conditions' and {n_ops} 'logical_operators'"
            raise ValueError(msg)

        for el in conditions:
            operator = el["operator"]
            value = el.get("value")
            compare_col = el.get("comparison_column")
            ensure_spark_condition(operator, value, compare_col)

        self._conditions: List[dict] = conditions
        self._assembly = iter([self._operators[i.lower()] for i in logical_operators])
        self._output_col: str = output_col

    def _transform(self, df):
        spark_conds = []
        for el in self._conditions:
            cond = get_spark_condition(
                df,
                el["column"],
                el["operator"],
                value=el.get("value"),
                compare_col=el.get("comparison_column"),
            )
            spark_conds.append(cond)

        # pylint: disable=unnecessary-lambda
        main_cond = reduce(lambda a, b: next(self._assembly)(a, b), spark_conds)
        main_cond = null_cond_to_false(main_cond)
        return df.withColumn(self._output_col, main_cond)


class MultipleLiterals(Transformer):
    def __init__(self, *, values: Dict[str, Dict[str, Any]]):
        """Assign scalar literals to new columns.

        If a specified column already exists, it will be overwritten with the new value.

        Args:
            values (dict[str, dict[str, any]]):
                A dictionary containing the column names and their
                corresponding values to be assigned. Example:
                {
                    "department": {"value": "finance"},
                    "employees": {"value": 10, "cast": "bigint"},
                    "active": {"value": True, "cast": "boolean"},
                }
                The "value" key is required, while the "cast" key is optional.
        """
        self._validate(values)
        super().__init__()
        self._values: Dict[str, Dict[str, str]] = values

    @staticmethod
    def _validate(values: Dict[str, Dict[str, str]]):
        allowed_keys = {"value", "cast"}

        for name, nd in values.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"Column name should be a <string>, found {name} <{type(name)}>"
                )
            if not isinstance(nd, dict):
                raise AssertionError(
                    f"Inner value for column specification must be <dict>. Found {name} <{type(nd)}"
                )
            keys = set(nd.keys())
            if not keys.issubset(allowed_keys):
                raise AssertionError(
                    f"Allowed keys for column specification: {allowed_keys}. Found {keys}"
                )
            cast = nd.get("cast")
            if cast:
                if not isinstance(cast, (str, DataType)):
                    msg = "If 'cast' is provided it must be <string> or <pyspark.sql.type.DataType>."
                    raise AssertionError(f"{msg} Found {name} <{type(cast)}")

    def _transform(self, df):
        cur_cols = [i for i in df.columns if i not in self._values]
        new_cols = []
        for name, nd in self._values.items():
            new_col = F.lit(nd["value"]).alias(name)
            cast = nd.get("cast")
            if cast:
                new_col = new_col.cast(cast)
            new_cols.append(new_col)

        return df.select(cur_cols + new_cols)


class ReplaceWithMap(Transformer):
    def __init__(
        self, *, input_col: str, replace: dict, output_col: Optional[str] = None
    ):
        """Replace the column values using the provided dictionary.

        Input and output types must be the same. All the values that are not
        mapped in "replace" remain untouched. None/null values are not
        allowed as keys in the mapping dictionary.

        Args:
            input_col (str):
                Input column.
            replace (dict):
                Dictionary for replacing.
            output_col (str | None):
                If not provided, the replacement takes place in the input_col.
        """
        _assert_no_null_keys(replace)
        super().__init__()
        self._input_col: str = input_col
        self._map: dict = replace
        self._output_col: Optional[str] = output_col

    def _transform(self, df):
        if self._output_col:
            output_col = self._output_col
            df = df.withColumn(output_col, F.col(self._input_col))
        else:
            output_col = self._input_col

        return df.replace(to_replace=self._map, subset=[output_col])


class RowWiseGreatestOrLeast(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        output_col: str,
        operation: str,
    ):
        """Create a new column with the row-wise greatest/least value of the input columns.

        NaN will be treated as null for the following types:
        - ‘FloatType’
        - 'DecimalType'
        - ‘DoubleType’

        Args:
            columns (list(str) | None):
                A list of columns to select. Defaults to None.
            regex (str | None):
                Take the columns to select by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Take the columns to select by using a bash-like pattern.
                Defaults to None.
            output_col (str):
                The name of the output column that will contain the row-wise
                greatest or least value.
            operation (str):
                The operation to perform. Must be either 'least' or 'greatest'.

        Raises:
            ValueError: If `operation` is not 'least' or 'greatest'.
            ValueError: If both `input_cols` and `input_regex` are not provided.
            ValueError: If no matching columns are found for the given input
                columns or regex.
        """
        assert_allowed(operation, {"least", "greatest"}, "operation")

        assert_at_least_one_non_null(columns, regex, glob)

        if columns:
            if isinstance(columns, str) or len(set(columns)) < 2:
                msg = "columns must be a list of unique strings "
                msg += "with 2 or more elements."
                raise AssertionError(msg)

        super().__init__()
        self._set_columns_selections(columns=columns, regex=regex, glob=glob)
        self._output_col = output_col
        self._operation = operation

    def _transform(self, df):
        cols: List[str] = self._get_selected_columns(df)

        if not cols:
            raise ValueError("no columns matched columns in the DF")

        types = set(df.schema[col].dataType for col in cols)
        if len(types) > 1:
            raise TypeError(f"Input columns have mixed data types. Found: {types}")

        df_selection = df.select(cols)
        clean_cols: List[Union[str, F.col]] = []
        for c in cols:
            data_type_name: str = get_column_data_type_name(df_selection, c)
            if data_type_name in {"decimal", "double", "float"}:
                clause = F.when(F.isnan(c), F.lit(None)).otherwise(F.col(c))
                clean_cols.append(clause.alias(c))
            else:
                clean_cols.append(c)

        func = {"greatest": F.greatest(*clean_cols), "least": F.least(*clean_cols)}
        return df.withColumn(self._output_col, func[self._operation])


class SqlFunction(Transformer):
    def __init__(
        self,
        *,
        column: str,
        function: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Call a pyspark.sql.function with the provided args/kwargs.

        Args:
            column (str):
                Name of the column where the result of the function is stored.
            function (str):
                Name of the pyspark.sql.function to call.
            args (list(any) | None):
                Positional arguments of pyspark.sql.function. Defaults to None.
            kwargs (dict(str, any) | None):
                Keyword arguments of pyspark.sql.function. Defaults to None.
        """
        valid_funcs = {
            i for i in dir(F) if (not i.startswith("_")) and (not i[0].isupper())
        }
        assert_allowed(function, valid_funcs, "function")

        super().__init__()
        self._output_col: str = column
        self._func_name: str = function
        self._args: list = args if args else []
        self._kwargs: Dict[str, Any] = kwargs if kwargs else {}

    def _transform(self, df):
        func = getattr(F, self._func_name)(*self._args, **self._kwargs)
        return df.withColumn(self._output_col, func)


class ToDF(Transformer):
    def __init__(self, *, columns: Optional[Union[str, List[str]]] = None):
        """Returns a new DataFrame that with new specified column names.

        Args:
            columns (list(str) | None):
                If passed, an iterable of strings representing the new column
                names. The length of the list needs to be the same as the
                number of columns in the initial DataFrame.
                If not passed, it uses all the columns in the initial DataFrame.
                Defaults to None.
        """
        super().__init__()
        self._cols: Optional[List[str]] = ensure_flat_list(columns) if columns else None

    def _transform(self, df):
        columns: List[str] = self._cols if self._cols else df.columns
        return df.toDF(*columns)


class UnionByName(Transformer):
    def __init__(
        self,
        *,
        temp_view: Optional[str] = None,
        store_key: Optional[str] = None,
        select_before_union: Optional[Union[str, List[str]]] = None,
        drop_before_union: Optional[Union[str, List[str]]] = None,
        drop_excess_columns: bool = False,
        allow_missing_columns: bool = False,
    ):
        """Append a dataframe to the main one in the pipeline.

        This dataframe can be retrieved either from Spark temporary
        views or Nebula storage.

        Args:
            temp_view (str | None):
                Dataframe name in Spark temporary views.
            store_key (str | None):
                Dataframe name in Nebula storage.
            select_before_union (str | list(str) | None):
                Columns to select in the dataframe to append before
                performing the union.
            drop_before_union (str | list(str) | None):
                Columns to drop in the dataframe to append before
                performing the union.
            drop_excess_columns (bool):
                If True, drop columns in the dataframe to append that are
                not present in the main dataframe.
                Defaults to False.
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
        if (bool(temp_view) + bool(store_key)) != 1:
            msg = "Either 'store_key' or 'temp_view' must be provided, but not both."
            raise ValueError(msg)

        assert_at_most_one_args(drop_excess_columns, allow_missing_columns)

        super().__init__()
        self._temp_view: Optional[str] = temp_view
        self._store_key: Optional[str] = store_key
        self._to_select: List[str] = ensure_flat_list(select_before_union)
        self._to_drop: List[str] = ensure_flat_list(drop_before_union)
        self._drop_excess_columns: bool = drop_excess_columns
        self._allow_missing_columns: bool = allow_missing_columns

    def __read(self, df):
        if self._temp_view:
            ss = get_spark_session(df)
            df_union = ss.table(self._temp_view)
        else:
            df_union = ns.get(self._store_key)

        if self._to_drop:
            df_union = df_union.drop(*self._to_drop)

        elif self._drop_excess_columns:
            input_cols = set(df.columns)
            to_drop = [i for i in df_union.columns if i not in input_cols]
            df_union = df_union.drop(*to_drop)

        if self._to_select:
            df_union = df_union.select(*self._to_select)

        return df_union

    def _transform(self, df):
        df_union = self.__read(df)

        # keep the if / else as spark 3.0.0 does not accept the 2nd parameter
        if self._allow_missing_columns:
            ret = df.unionByName(df_union, allowMissingColumns=True)
        else:
            ret = df.unionByName(df_union)
        return ret


class When(Transformer):
    def __init__(
        self,
        *,
        output_column: str,
        conditions: List[Dict],
        otherwise_constant: Any = None,
        otherwise_column: Optional[str] = None,
        cast_output: Optional[str] = None,
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

        self._conditions: List[Dict[str, Any]] = conditions
        self._cast_output: Optional[str] = cast_output

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
