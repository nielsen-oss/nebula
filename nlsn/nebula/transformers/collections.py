from typing import Iterable

import narwhals as nw
import pandas as pd

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.auxiliaries import assert_allowed
from nlsn.nebula.base import Transformer

__all__ = [
    "AppendDataFrame",
    "DropNulls",
    "InjectData",
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
        if isinstance(df_native, pd.DataFrame):
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
