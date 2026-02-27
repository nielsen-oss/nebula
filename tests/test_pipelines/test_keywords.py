"""Test pipeline loader keywords."""

import numpy as np
import polars as pl
import pytest

from nebula import TransformerPipeline
from nebula.pipelines.pipeline_loader import load_pipeline
from nebula.storage import nebula_storage as ns
from nebula.transformers import DropNulls

from ..auxiliaries import pl_assert_equal


def _build_pipe(data, source: str) -> TransformerPipeline:
    """Build a single-step pipeline with the given string keyword."""
    if source == "python":
        return TransformerPipeline([data])
    else:
        return load_pipeline({"pipeline": [data]})


def _native_type(df) -> type:
    """Unwrap narwhals and return the native Python type."""
    import narwhals as nw

    if isinstance(df, (nw.DataFrame, nw.LazyFrame)):
        return type(nw.to_native(df))
    return type(df)


@pytest.mark.parametrize("source", ["python", "yaml"])
def test_from_store(source: str):
    """Test 'from_store' functionality."""
    df_input = pl.DataFrame({"c1": np.arange(20)})
    ns.clear()
    df2 = pl.concat([df_input, df_input], how="vertical")
    store_data = {"from_store": "df2"}
    pipe = _build_pipe(store_data, source)
    pipe.show()

    try:
        ns.set("df2", df2)
        df_out = pipe.run(df_input)
        pl_assert_equal(df_out, df2)
    finally:
        ns.clear()


@pytest.mark.parametrize("source", ["python", "yaml"])
def test_store(source: str):
    """Test 'store' functionality."""
    df_input = pl.DataFrame({"a": [1, None, 2]})
    ns.clear()

    if source == "python":
        pipe = TransformerPipeline([DropNulls(glob="*"), {"store": "df_no_null"}])
    else:
        pipe_text = [
            {"transformer": "DropNulls", "params": {"glob": "*"}},
            {"store": "df_no_null"},
        ]
        pipe = load_pipeline({"pipeline": pipe_text})

    pipe.show()

    try:
        df_out = pipe.run(df_input)
        df_no_null = ns.get("df_no_null")
        pl_assert_equal(df_out, df_no_null)
    finally:
        ns.clear()


@pytest.mark.parametrize("debug_active", [True, False])
@pytest.mark.parametrize("source", ["python", "yaml"])
def test_store_debug(debug_active: bool, source: str):
    """Test 'store_debug' functionality."""
    df_input = pl.DataFrame({"a": [1, None, 2]})
    ns.clear()
    ns.allow_debug(debug_active)

    if source == "python":
        pipe = TransformerPipeline([DropNulls(glob="*"), {"store_debug": "df_no_null"}])
    else:
        pipe_text = [
            {"transformer": "DropNulls", "params": {"glob": "*"}},
            {"store_debug": "df_no_null"},
        ]
        pipe = load_pipeline({"pipeline": pipe_text})

    pipe.show()

    try:
        df_out = pipe.run(df_input)
        if debug_active:
            df_no_null = ns.get("df_no_null")
            pl_assert_equal(df_out, df_no_null)
        else:
            with pytest.raises(KeyError):
                ns.get("df_no_null")
    finally:
        ns.clear()
        ns.allow_debug(False)


@pytest.mark.parametrize("source", ["python", "yaml"])
class TestCollectKeyword:
    """Pipeline integration tests for the 'collect' keyword."""

    DATA = {"a": [1, 2, 3], "b": [4, 5, 6]}

    def test_lazyframe_becomes_dataframe(self, source: str):
        pipe = _build_pipe("collect", source)
        result = pipe.run(pl.LazyFrame(self.DATA))
        assert _native_type(result) is pl.DataFrame

    def test_lazyframe_result_is_not_lazy(self, source: str):
        pipe = _build_pipe("collect", source)
        result = pipe.run(pl.LazyFrame(self.DATA))
        assert not isinstance(result, pl.LazyFrame)

    def test_lazyframe_data_is_preserved(self, source: str):
        pipe = _build_pipe("collect", source)
        result = pipe.run(pl.LazyFrame(self.DATA))
        pl.testing.assert_frame_equal(result, pl.DataFrame(self.DATA))

    def test_dataframe_stays_dataframe(self, source: str):
        pipe = _build_pipe("collect", source)
        result = pipe.run(pl.DataFrame(self.DATA))
        assert _native_type(result) is pl.DataFrame

    def test_dataframe_noop_data_is_preserved(self, source: str):
        pipe = _build_pipe("collect", source)
        result = pipe.run(pl.DataFrame(self.DATA))
        pl.testing.assert_frame_equal(result, pl.DataFrame(self.DATA))

    def test_pandas_raises_type_error(self, source: str):
        import pandas as pd

        pipe = _build_pipe("collect", source)
        with pytest.raises(TypeError, match="collect"):
            pipe.run(pd.DataFrame(self.DATA))

    def test_pipe_show_does_not_raise(self, source: str):
        _build_pipe("collect", source).show()


@pytest.mark.parametrize("source", ["python", "yaml"])
class TestToLazyKeyword:
    """Pipeline integration tests for the 'to_lazy' keyword."""

    DATA = {"a": [1, 2, 3], "b": [4, 5, 6]}

    def test_dataframe_becomes_lazyframe(self, source: str):
        pipe = _build_pipe("to_lazy", source)
        result = pipe.run(pl.DataFrame(self.DATA))
        assert _native_type(result) is pl.LazyFrame

    def test_dataframe_result_is_not_eager(self, source: str):
        pipe = _build_pipe("to_lazy", source)
        result = pipe.run(pl.DataFrame(self.DATA))
        assert not isinstance(result, pl.DataFrame)

    def test_dataframe_data_is_preserved(self, source: str):
        pipe = _build_pipe("to_lazy", source)
        result = pipe.run(pl.DataFrame(self.DATA))
        pl.testing.assert_frame_equal(result.collect(), pl.DataFrame(self.DATA))

    def test_lazyframe_stays_lazyframe(self, source: str):
        pipe = _build_pipe("to_lazy", source)
        result = pipe.run(pl.LazyFrame(self.DATA))
        assert _native_type(result) is pl.LazyFrame

    def test_lazyframe_noop_data_is_preserved(self, source: str):
        pipe = _build_pipe("to_lazy", source)
        result = pipe.run(pl.LazyFrame(self.DATA))
        pl.testing.assert_frame_equal(result.collect(), pl.DataFrame(self.DATA))

    def test_pandas_raises_type_error(self, source: str):
        import pandas as pd

        pipe = _build_pipe("to_lazy", source)
        with pytest.raises(TypeError, match="to_lazy"):
            pipe.run(pd.DataFrame(self.DATA))

    def test_pipe_show_does_not_raise(self, source: str):
        _build_pipe("to_lazy", source).show()
