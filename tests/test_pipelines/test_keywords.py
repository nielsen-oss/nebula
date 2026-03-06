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
    df2 = pl.concat([df_input, df_input], how="vertical")
    pipe = _build_pipe({"from_store": "df2"}, source)

    ns.set("df2", df2)
    df_out = pipe.run(df_input)
    pl_assert_equal(df_out, df2)


@pytest.mark.parametrize("source", ["python", "yaml"])
def test_store(source: str):
    """Test 'store' functionality."""
    df_input = pl.DataFrame({"a": [1, None, 2]})

    if source == "python":
        pipe = TransformerPipeline([DropNulls(glob="*"), {"store": "df_no_null"}])
    else:
        pipe_text = [
            {"transformer": "DropNulls", "params": {"glob": "*"}},
            {"store": "df_no_null"},
        ]
        pipe = load_pipeline({"pipeline": pipe_text})

    df_out = pipe.run(df_input)
    pl_assert_equal(df_out, ns.get("df_no_null"))


@pytest.mark.parametrize("debug_active", [True, False])
@pytest.mark.parametrize("source", ["python", "yaml"])
def test_store_debug(debug_active: bool, source: str):
    """Test 'store_debug' functionality."""
    df_input = pl.DataFrame({"a": [1, None, 2]})
    ns.allow_debug(debug_active)

    if source == "python":
        pipe = TransformerPipeline([DropNulls(glob="*"), {"store_debug": "df_no_null"}])
    else:
        pipe_text = [
            {"transformer": "DropNulls", "params": {"glob": "*"}},
            {"store_debug": "df_no_null"},
        ]
        pipe = load_pipeline({"pipeline": pipe_text})

    df_out = pipe.run(df_input)

    if debug_active:
        pl_assert_equal(df_out, ns.get("df_no_null"))
    else:
        with pytest.raises(KeyError):
            ns.get("df_no_null")

    ns.allow_debug(False)


@pytest.mark.parametrize("source", ["python", "yaml"])
@pytest.mark.parametrize("keyword", ["collect", "to_lazy"])
class TestCollectAndToLazyKeywords:
    """Pipeline integration tests for 'collect' and 'to_lazy' keywords."""

    DATA = {"a": [1, 2, 3], "b": [4, 5, 6]}

    def _expected_type(self, keyword: str) -> type:
        return pl.DataFrame if keyword == "collect" else pl.LazyFrame

    def _opposite_type(self, keyword: str) -> type:
        return pl.LazyFrame if keyword == "collect" else pl.DataFrame

    def _input_of_opposite_type(self, keyword: str):
        """Input that triggers the conversion (opposite type)."""
        if keyword == "collect":
            return pl.LazyFrame(self.DATA)
        return pl.DataFrame(self.DATA)

    def _input_of_same_type(self, keyword: str):
        """Input already in target type (noop)."""
        if keyword == "collect":
            return pl.DataFrame(self.DATA)
        return pl.LazyFrame(self.DATA)

    def _to_comparable(self, result):
        """Collect lazy results for comparison."""
        if isinstance(result, pl.LazyFrame):
            return result.collect()
        return result

    def test_converts_to_target_type(self, source: str, keyword: str):
        pipe = _build_pipe(keyword, source)
        result = pipe.run(self._input_of_opposite_type(keyword))
        assert _native_type(result) is self._expected_type(keyword)

    def test_result_is_not_opposite_type(self, source: str, keyword: str):
        pipe = _build_pipe(keyword, source)
        result = pipe.run(self._input_of_opposite_type(keyword))
        assert not isinstance(result, self._opposite_type(keyword))

    def test_data_is_preserved_after_conversion(self, source: str, keyword: str):
        pipe = _build_pipe(keyword, source)
        result = pipe.run(self._input_of_opposite_type(keyword))
        pl.testing.assert_frame_equal(self._to_comparable(result), pl.DataFrame(self.DATA))

    def test_same_type_stays_same(self, source: str, keyword: str):
        pipe = _build_pipe(keyword, source)
        result = pipe.run(self._input_of_same_type(keyword))
        assert _native_type(result) is self._expected_type(keyword)

    def test_noop_data_is_preserved(self, source: str, keyword: str):
        pipe = _build_pipe(keyword, source)
        result = pipe.run(self._input_of_same_type(keyword))
        pl.testing.assert_frame_equal(self._to_comparable(result), pl.DataFrame(self.DATA))

    def test_pandas_raises_type_error(self, source: str, keyword: str):
        import pandas as pd

        pipe = _build_pipe(keyword, source)
        with pytest.raises(TypeError, match=keyword):
            pipe.run(pd.DataFrame(self.DATA))

    def test_pipe_show_does_not_raise(self, source: str, keyword: str):
        _build_pipe(keyword, source)
