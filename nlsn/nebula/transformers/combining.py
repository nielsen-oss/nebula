"""Combining multiple DataFrames."""

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.base import Transformer
from nlsn.nebula.nw_util import append_dataframes, assert_join_params, join_dataframes

__all__ = ["AppendDataFrame", "Join"]


class AppendDataFrame(Transformer):
    def __init__(
            self,
            *,
            store_key: str | None = None,
            allow_missing_cols: bool = False,
            relax: bool = False,
            rechunk: bool = False,
            ignore_index: bool = False,
    ):
        """Append a dataframe to the main one in the pipeline.

        Args:
            store_key (str | None):
                Dataframe name in Nebula storage.
            allow_missing_cols (bool):
                If True, allows column mismatches between dataframes. Missing columns
                are filled with null values. If False, raises ValueError when column
                sets don't match exactly. Defaults to False.
                Behavior by backend:
                - pandas: Uses pd.concat naturally handles missing columns
                - Polars: Uses 'diagonal' mode to add null columns
                - Spark: Uses unionByName(allowMissingColumns=True)
            relax (bool):
                Polars-only parameter. If True, allows compatible type coercion during
                concatenation (e.g., int32 → int64, float32 → float64). Uses Polars'
                'vertical_relaxed' or 'diagonal_relaxed' modes. Ignored for pandas and
                Spark. Defaults to False.
            rechunk (bool):
                Polars-only parameter. If True, rechunks the concatenated result for
                better memory layout and performance. Ignored for pandas and Spark.
                Defaults to False.
            ignore_index (bool):
                Pandas-only parameter. If True, do not preserve the original index
                values when concatenating. Ignored for Polars (no index) and Spark.
                Defaults to False.
        """
        super().__init__()
        self._store_key: str | None = store_key
        self._kws = {
            "allow_missing_cols": allow_missing_cols,
            "relax": relax,
            "rechunk": rechunk,
            "ignore_index": ignore_index,
        }

    def transform(self, df):
        df_union = ns.get(self._store_key)
        return append_dataframes([df, df_union], **self._kws)


class Join(Transformer):
    def __init__(
            self,
            *,
            store_key: str,
            how: str,
            on: list[str] | str | None = None,
            left_on: str | list[str] | None = None,
            right_on: str | list[str] | None = None,
            suffix: str = "_right",
            broadcast: bool = False
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
            broadcast (bool):
                Spark-only parameter. If True, broadcast the right df.
                Ignored for Pandas and Polars. Defaults to False.
        """
        assert_join_params(how, on, left_on, right_on)

        super().__init__()
        self._store_key: str = store_key
        self._join_kwargs = {
            "how": how,
            "on": on,
            "left_on": left_on,
            "right_on": right_on,
            "suffix": suffix,
            "broadcast": broadcast,
        }

    def _transform_nw(self, df):
        df_to_join = ns.get(self._store_key)
        return join_dataframes(df, df_to_join, **self._join_kwargs)
