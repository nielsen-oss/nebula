"""General Purposes Transformers."""

from functools import reduce
from typing import Any

from pyspark.sql import functions as F

from nlsn.nebula.auxiliaries import (
    assert_at_least_one_non_null,
    assert_at_most_one_args,
    ensure_flat_list,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_util import (
    ensure_spark_condition,
    get_spark_condition,
    get_spark_session,
)
from nlsn.nebula.storage import nebula_storage as ns

__all__ = [
    "FillNa",
    "Join",
    "Melt",
    "UnionByName",
    "When",
]


class FillNa(Transformer):
    def __init__(
            self,
            *,
            value: int | float | str | bool | dict[str, int | float | str | bool],
            columns: str | list[str] | None = None,
            regex: str | None = None,
            glob: str | None = None,
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

        scalar = int | float | str | bool
        self._value: scalar | dict[str, scalar] = value
        self._set_columns_selections(columns=columns, regex=regex, glob=glob)

    def _transform(self, df):
        if self._flag_mapping:
            return df.na.fill(self._value)
        subset = self._get_selected_columns(df)
        return df.na.fill(self._value, subset=subset)


class Join(Transformer):
    def __init__(
            self,
            *,
            table: str,
            on: list[str] | str,
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
        self._on: list[str] = ensure_flat_list(on)
        self._how: str = how
        self._broadcast: bool = broadcast

    def _transform(self, df):
        df_right = ns.get(self._table)
        if self._broadcast:
            df_right = F.broadcast(df_right)
        return df.join(df_right, on=self._on, how=self._how)


class Melt(Transformer):
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
        id_cols: list[str] = self._get_selected_columns(df)

        self._set_columns_selections(
            columns=self._melt_cols,
            regex=self._melt_regex,
        )
        melt_cols: list[str] = self._get_selected_columns(df)

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


class UnionByName(Transformer):
    def __init__(
            self,
            *,
            temp_view: str | None = None,
            store_key: str | None = None,
            select_before_union: str | list[str] | None = None,
            drop_before_union: str | list[str] | None = None,
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
        self._temp_view: str | None = temp_view
        self._store_key: str | None = store_key
        self._to_select: list[str] = ensure_flat_list(select_before_union)
        self._to_drop: list[str] = ensure_flat_list(drop_before_union)
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
