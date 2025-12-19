"""Test 'skip' / 'perform' pipeline functionalities."""

from copy import deepcopy

import polars as pl
import pytest

from nebula.pipelines.pipeline_loader import load_pipeline
from nebula.pipelines.pipelines import TransformerPipeline
from nebula.transformers import AddLiterals
from ..auxiliaries import pl_assert_equal


@pytest.mark.parametrize("skip, perform", ([True, None], [None, False]))
class TestSkipPipeline:
    trf_py = AddLiterals(data=[{"value": "x", "alias": "c2"}])
    trf_text = {"transformer": "AddLiterals", "params": {"data": {"value": "x", "alias": "c2"}}}
    df_input = pl.DataFrame({"c1": [1, 2]})

    def test_py_flat_pipeline(self, skip, perform):
        pipe = TransformerPipeline(self.trf_py, skip=skip, perform=perform)
        df_chk = pipe.run(self.df_input)
        pl_assert_equal(self.df_input, df_chk)

    def test_py_split_pipeline(self, skip, perform):
        pipe = TransformerPipeline(
            {
                "c_x": self.trf_py,
                "c_y": self.trf_py,
            },
            split_function=lambda x: x,  # not called actually
            skip=skip,
            perform=perform,
        )
        df_chk = pipe.run(self.df_input)
        pl_assert_equal(self.df_input, df_chk)

    def test_text_flat_pipeline(self, skip, perform):
        data = deepcopy(self.trf_text)
        data["skip"] = skip
        data["perform"] = perform
        df_chk = load_pipeline({"pipeline": data}).run(self.df_input)
        pl_assert_equal(self.df_input, df_chk)

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
    data = {"transformer": "AddLiterals", "params": {"data": {"value": "x", "alias": "c2"}}}
    if skip is not None:
        data["skip"] = skip
    if perform is not None:
        data["perform"] = perform

    df_input = pl.DataFrame({"c1": [1, 2]})
    df_chk = load_pipeline({"pipeline": data}).run(df_input)
    pl_assert_equal(df_input, df_chk)
