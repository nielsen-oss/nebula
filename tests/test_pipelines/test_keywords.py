"""Test pipeline loader keywords."""

import numpy as np
import polars as pl
import pytest

from nebula.pipelines.pipeline_loader import load_pipeline
from nebula.pipelines.pipelines import TransformerPipeline
from nebula.storage import nebula_storage as ns
from nebula.transformers import DropNulls
from ..auxiliaries import pl_assert_equal


@pytest.mark.parametrize("source", ["python", "yaml"])
def test_replace_with_stored_df(source: str):
    """Test 'replace_with_stored_df' functionality."""
    df_input = pl.DataFrame({"c1": np.arange(20)})
    ns.clear()
    df2 = pl.concat([df_input, df_input], how="vertical")
    store_data = {"replace_with_stored_df": "df2"}

    if source == "python":
        pipe = TransformerPipeline([store_data])
    else:
        pipe = load_pipeline({"pipeline": store_data})

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
            {"store": "df_no_null"}
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
            {"store_debug": "df_no_null"}
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
