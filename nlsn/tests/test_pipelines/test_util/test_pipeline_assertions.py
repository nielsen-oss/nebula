"""Catch pipeline assertion errors."""

import pytest

from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.transformers import SelectColumns


def test_pipeline_wrong_data_type():
    """Test TransformerPipeline providing wrong data type."""
    with pytest.raises(TypeError):
        TransformerPipeline({"1", "2"})


def test_pipeline_branch_and_apply_to_rows():
    """Test TransformerPipeline providing both 'branch' and 'apply_to_rows'."""
    with pytest.raises(AssertionError):
        TransformerPipeline(
            SelectColumns(columns="c1"),
            branch={"end": "append"},
            apply_to_rows={"input_col": "x", "operator": "isNull"},
        )


@pytest.mark.parametrize(
    "branch, apply_to_rows",
    [
        ({"end": "dead-end", "storage": "x"}, None),
        (None, {"input_col": "x", "operator": "isNull"}),
    ],
)
def test_branch_and_apply_to_rows_in_split_pipeline(branch, apply_to_rows):
    """Test a split pipeline providing the 'branch' / 'and_apply_to_rows'."""
    d = {
        "errors": SelectColumns(columns="c1"),
        "valid_problems": [SelectColumns(columns="c2")],
    }
    with pytest.raises(AssertionError):
        TransformerPipeline(
            d, split_function=lambda x: x, branch=branch, apply_to_rows=apply_to_rows
        )


@pytest.mark.parametrize(
    "split_order", [["low"], ["hi"], ["hi", "low", "wrong"], ["hi", "low1"]]
)
def test_split_pipeline_wrong_split_order(split_order):
    """Test TransformerPipeline providing wrong 'split_order' parameters."""
    data = {"low": [], "hi": []}
    with pytest.raises(KeyError):
        TransformerPipeline(data, split_function=lambda x: x, split_order=split_order)


def test_split_pipeline_wrong_split_type():
    """Test TransformerPipeline providing wrong split type (int)."""
    data = {"low": [], 1: []}
    with pytest.raises(TypeError):
        TransformerPipeline(data, split_function=lambda x: x)


def test_split_pipeline_wrong_split_order_type():
    """Test TransformerPipeline providing a wrong split_order type."""
    data = {"low": [], "1": []}
    split_order = ["low", 1]
    with pytest.raises(TypeError):
        TransformerPipeline(data, split_function=lambda x: x, split_order=split_order)


@pytest.mark.parametrize("func, err", [(set(), TypeError), (None, AssertionError)])
def test_split_pipeline_wrong_split_function(func, err):
    """Test TransformerPipeline providing a wrong split_function type."""
    with pytest.raises(err):
        TransformerPipeline({"low": [], "1": []}, split_function=func)
