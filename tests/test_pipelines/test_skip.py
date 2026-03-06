"""Test 'skip' / 'perform' pipeline functionalities."""

import polars as pl
import pytest

from nebula import TransformerPipeline
from nebula.pipelines.pipeline_loader import load_pipeline
from nebula.transformers import AddLiterals

from ..auxiliaries import pl_assert_equal

_TRF_PY = AddLiterals(data=[{"value": "x", "alias": "c2"}])
_DF_INPUT = pl.DataFrame({"c1": [1, 2]})


def _trf_text():
    return {
        "transformer": "AddLiterals",
        "params": {"data": {"value": "x", "alias": "c2"}},
    }


@pytest.mark.parametrize("skip, perform", ([True, None], [None, False]))
class TestSkipPipeline:
    def test_py_flat_pipeline(self, skip, perform):
        pipe = TransformerPipeline(_TRF_PY, skip=skip, perform=perform)
        df_chk = pipe.run(_DF_INPUT)
        pl_assert_equal(_DF_INPUT, df_chk)

    def test_py_split_pipeline(self, skip, perform):
        pipe = TransformerPipeline(
            {
                "c_x": _TRF_PY,
                "c_y": _TRF_PY,
            },
            split_function=lambda x: x,  # not called actually
            skip=skip,
            perform=perform,
        )
        df_chk = pipe.run(_DF_INPUT)
        pl_assert_equal(_DF_INPUT, df_chk)

    def test_text_flat_pipeline(self, skip, perform):
        data = _trf_text()
        data["skip"] = skip
        data["perform"] = perform
        df_chk = load_pipeline({"pipeline": data}).run(_DF_INPUT)
        pl_assert_equal(_DF_INPUT, df_chk)

    @pytest.mark.parametrize("data", [{"transformer": "invalid"}, {"wrong_key": "invalid"}])
    def test_text_invalid_arguments(self, data, skip, perform):
        """Ensures that a skipped pipeline does not attempt to parse its arguments.

        This is useful when the pipeline is skipped and its arguments might be
        incomplete or invalid if the pipeline is not executed. By skipping the
        argument-parsing step, we avoid potential / unnecessary errors.
        """
        data = {"skip": skip, "perform": perform, "pipeline": data}
        load_pipeline(data)


@pytest.mark.parametrize("skip, perform", ([True, None], [None, False]))
def test_skip_transformer(skip, perform):
    """Test 'skip' / 'perform' transformer functionality."""
    data = _trf_text()
    if skip is not None:
        data["skip"] = skip
    if perform is not None:
        data["perform"] = perform

    df_chk = load_pipeline({"pipeline": data}).run(_DF_INPUT)
    pl_assert_equal(_DF_INPUT, df_chk)
