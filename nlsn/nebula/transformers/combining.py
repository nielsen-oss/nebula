"""Combining multiple DataFrames."""

import narwhals as nw

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.auxiliaries import assert_allowed, ensure_flat_list
from nlsn.nebula.base import Transformer
from nlsn.nebula.df_types import get_dataframe_type

__all__ = ["AppendDataFrame", "Join"]


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
        if get_dataframe_type(df_native) == "pandas":
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
