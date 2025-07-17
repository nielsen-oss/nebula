"""Data Cleanup and Fixing Utilities."""

from functools import reduce
from operator import and_
from typing import Iterable, List, Optional, Union

import pyspark.sql.functions as F
from pyspark.sql.types import DecimalType, DoubleType, FloatType, StructField
from pyspark.sql.window import Window

from nlsn.nebula.auxiliaries import (
    assert_allowed,
    assert_is_numeric,
    assert_is_string,
    assert_only_one_non_none,
    ensure_flat_list,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.spark_util import ensure_spark_condition, get_spark_condition

__all__ = [
    "ClipOrNull",
    "FillColumnWithMode",
    "NanToNull",
]


class ClipOrNull(Transformer):
    def __init__(
        self,
        *,
        input_col: str,
        operator: str,
        perform: str,
        ref_value: Union[int, float] = None,
        output_col: Optional[str] = None,
        comparison_col: Optional[str] = None,
    ):
        """Clip or null to a certain value using "lt" or "gt" operator.

        `ref_value` or `comparison_column` are mutually exclusive.

        If the input value in the `input_col` is null or NaN, the output
        value will be a null value, NaN values are never returned.

        Args:
            input_col (str):
                Input column.
            operator (str):
                "lt" (lower than) or "gt" (greater than).
            perform (str):
                "clip": Replace with ref_value, "null": replace with null.
            ref_value (int | float | None):
                Value to compare with. Defaults to None.
            output_col (str | None):
                Name of the output column. If not provided, it will replace
                the input_col. Defaults to None.
            comparison_col (str | None):
                Name of column to be compared with input_col.
                For all null / nan values in 'comparison_col' the outcome will
                be None. Defaults to None.
        """
        operator = operator.lower()
        perform = perform.lower()
        assert_allowed(operator, {"lt", "gt"}, "operator")
        assert_allowed(perform, {"clip", "null"}, "perform")
        assert_only_one_non_none(ref_value, comparison_col)

        if comparison_col:
            assert_is_string(comparison_col, "comparison_col")
        else:
            assert_is_numeric(ref_value, "ref_value")

        super().__init__()

        ensure_spark_condition(operator, ref_value, comparison_col)

        self._input_col: str = input_col
        self._operator: str = operator
        self._ref: Optional[Union[int, float]] = ref_value
        self._perform: str = perform
        self._cmp_col: Optional[str] = comparison_col
        self._output_col: Optional[str]
        self._output_col = input_col if output_col is None else output_col

    def _transform(self, df):
        cmp_cond = get_spark_condition(
            df,
            self._input_col,
            self._operator,
            value=self._ref,
            compare_col=self._cmp_col,
        )

        null_cond = F.col(self._input_col).isNull()
        null_cond |= F.isnan(self._input_col)

        input_value: F.col = F.col(self._input_col)

        if self._ref is not None:
            replace = F.lit(self._ref) if self._perform == "clip" else F.lit(None)
        else:
            null_cond |= F.col(self._cmp_col).isNull()
            null_cond |= F.isnan(self._cmp_col)
            replace = F.col(self._cmp_col) if self._perform == "clip" else F.lit(None)

        valid_clause = F.when(cmp_cond, replace).otherwise(input_value)
        out_clause = F.when(null_cond, F.lit(None)).otherwise(valid_clause)

        return df.withColumn(self._output_col, out_clause)


class FillColumnWithMode(Transformer):
    def __init__(
        self,
        *,
        to_fill_cols: Union[str, List[str]],
        groupby_columns: Union[str, List[str], None] = None,
        groupby_regex: Optional[str] = None,
        groupby_glob: Optional[str] = None,
        groupby_startswith: Optional[Union[str, Iterable[str]]] = None,
        groupby_endswith: Optional[Union[str, Iterable[str]]] = None,
        replace: bool = False,
        df_mode_to_pandas: bool = False,
    ):
        """Fill missing values with mode after grouping by the "groupby_cols".

        If the 'replace' parameter is True, all the values will be replaced
        with the mode, otherwise only the missing values are filled.

        It does *NOT* ensure that all values are filled, as if in a grouped
        set all the values are null, there are no valid values to use.

        Args:
            to_fill_cols (str | list(str)):
                Columns to be replaced with the mode.
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
            replace (bool):
                True to replace all the values with the mode, otherwise use
                the mode to fill null values only.
            df_mode_to_pandas (bool):
                Convert the df of the mode (small subset) to pandas before
                rejoining it into the original DF, to break the physical plan.

        Returns (pyspark.sql.DataFrame):
            DataFrame filled with the mode values in the "to_fill_cols" columns.
        """
        assert_only_one_non_none(groupby_columns, groupby_regex, groupby_glob)

        super().__init__()
        self._to_fill_cols: List[str] = ensure_flat_list(to_fill_cols)
        self._set_columns_selections(
            columns=groupby_columns,
            regex=groupby_regex,
            glob=groupby_glob,
            startswith=groupby_startswith,
            endswith=groupby_endswith,
        )
        self._replace: bool = replace
        self._df_mode_to_pandas: bool = df_mode_to_pandas

    def _transform(self, df):
        input_cols = df.columns
        groupby_cols = self._get_selected_columns(df)
        tot_cols = groupby_cols[:] + self._to_fill_cols

        # Create not null condition
        not_null_cond = [F.col(c).isNotNull() for c in tot_cols]

        # The windowing is done by count descending since we want the most
        # occurring value for that group.
        # Also, order by the 'groupby_cols' just for repeatability (for cases
        # with the same number of occurrences, where the mode could be more
        # than one value).
        win = Window.partitionBy(*groupby_cols).orderBy(
            F.desc("_count_"), *[F.desc(c) for c in tot_cols]
        )

        # Compute mode
        df_mode = (
            df.filter(reduce(and_, not_null_cond))
            .groupBy(*tot_cols)
            .agg(F.count("*").alias("_count_"))
            .withColumn("_row_number_", F.row_number().over(win))
            .filter(F.col("_row_number_") == 1)
            .select(
                *groupby_cols,
                *[F.col(c).alias(f"_mode_{c}_") for c in self._to_fill_cols],
            )
        )

        if self._df_mode_to_pandas:
            ss = df.sql_ctx.sparkSession
            schema = df_mode.schema
            logger.info("Converting df mode to pandas before rejoining ...")
            df_mode_pd = df_mode.toPandas()
            logger.info(f"df mode converted to pandas. Shape: {df_mode_pd.shape}")
            logger.info("Converting back df mode to spark ...")
            df_mode = ss.createDataFrame(df_mode_pd, schema=schema)

        # Add the mode of the column to the dataframe
        df = df.join(df_mode, on=groupby_cols, how="left")

        output_cols = []
        set_filled = set(self._to_fill_cols)
        for c in input_cols:
            if c in set_filled:
                if self._replace:
                    new_col = F.coalesce(F.col(f"_mode_{c}_"), F.col(c))
                else:
                    new_col = F.coalesce(F.col(c), F.col(f"_mode_{c}_"))

                output_cols.append(new_col.alias(c))
            else:
                output_cols.append(c)

        return df.select(*output_cols)


class NanToNull(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
    ):
        """Nan conversion to null.

        It is applicable exclusively to ‘FloatType’, 'DecimalType' and
        ‘DoubleType’.

        Args:
            columns (str | list(str) | None):
                A list of the objective columns. Defaults to None.
            regex (str | None):
                Select the objective columns by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the objective columns by using a bash-like pattern.
                Defaults to None.
        """
        super().__init__()
        self._set_columns_selections(columns=columns, regex=regex, glob=glob)

    def _transform(self, df):
        cols_to_parse: List[str] = self._get_selected_columns(df)
        new_cols = []

        field: StructField  # like StructField(column_name, FloatType, true)
        c: str
        for field in df.schema:
            c = field.name
            if c not in cols_to_parse:
                # do not parse
                new_cols.append(c)
                continue

            # Check only FloatType and DoubleType, the only types
            # that can have NaN according to the documentation.
            if isinstance(field.dataType, (FloatType, DecimalType, DoubleType)):
                clause = F.when(F.isnan(c), F.lit(None)).otherwise(F.col(c))
                new_cols.append(clause.alias(c))
            else:
                # "c" is an input column but cannot contain NaN
                new_cols.append(c)

        return df.select(*new_cols)
