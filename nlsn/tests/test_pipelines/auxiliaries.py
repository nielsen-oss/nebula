"""Auxiliaries for testing pipelines."""

import narwhals as nw
from polars.testing import assert_frame_equal

from nlsn.nebula.base import Transformer
from nlsn.nebula.storage import nebula_storage as ns

__all__ = [
    "pl_assert_equal",
    "AddOne",
    "CallMe",
    "Distinct",
    "RoundValues",
    "ThisTransformerIsBroken",
]


def pl_assert_equal(df_chk, df_exp) -> None:
    if isinstance(df_chk, (nw.DataFrame, nw.LazyFrame)):
        df_chk = nw.to_native(df_chk)
    if isinstance(df_exp, (nw.DataFrame, nw.LazyFrame)):
        df_exp = nw.to_native(df_exp)
    assert_frame_equal(df_chk, df_exp)


# =============================================================================
# Test Transformers (simple implementations for testing)
# =============================================================================

class AddOne(Transformer):

    def __init__(self, column: str):
        super().__init__()
        self._col: str = column

    def _transform_nw(self, df):
        if self._col in df.columns:
            value = df[self._col].max() + 1
        else:
            value = 0
        df = df.with_columns(nw.lit(value).alias(self._col))
        return df


class CallMe(Transformer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _transform_nw(df):
        if "_call_me_" in ns.list_keys():
            value = ns.get("_call_me_") + 1
            ns.set("_call_me_", value)
        else:
            ns.set("_call_me_", 1)
        return df


class Distinct(Transformer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _transform_nw(df):
        return df.unique()


class RoundValues(Transformer):

    def __init__(self, *, input_columns: str, precision: int):
        super().__init__()
        self._col = input_columns
        self._precision = precision

    def _transform_nw(self, df):
        import narwhals as nw
        return df.with_columns(
            nw.col(self._col).round(self._precision).alias(self._col)
        )


class ThisTransformerIsBroken(Transformer):
    @staticmethod
    def _transform_nw(df):
        raise ValueError("Broken transformer")
