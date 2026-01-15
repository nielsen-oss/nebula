"""Test the pipeline loader starting from YAML data with a dry-run."""

import pytest

from nebula.pipelines.pipeline_loader import (
    _extract_function,
    extract_lazy_params,
    load_pipeline,
)
from nebula.storage import nebula_storage as ns

from ..auxiliaries import load_yaml


def my_split_function(_df):
    """Mock split-function."""
    df_1 = ...
    df_2 = ...
    return {"split_1": df_1, "split_2": df_2}


exposed_functions = [my_split_function]


@pytest.mark.parametrize(
    "pipeline_key",
    [
        "wrong_split_name_1",
        "wrong_split_name_2",
        "wrong_pipeline",
    ],
)
def test_load_pipeline_wrong_split_pipelines(pipeline_key: str):
    """Test wrong split-pipelines."""
    file_name = "wrong_splits.yml"
    data = load_yaml(file_name)[pipeline_key]

    with pytest.raises(AssertionError):
        load_pipeline(data, extra_functions=exposed_functions)


@pytest.mark.parametrize("pipeline_key", ["pipeline_1"])
def test_load_pipeline_mock_dry_run(pipeline_key: str):
    """Test wrong split-pipelines."""
    file_name = "mock.yml"
    data = load_yaml(file_name)[pipeline_key]

    pipe = load_pipeline(data, extra_functions=exposed_functions)

    pipe.show(add_params=True)


def test_load_pipeline_wrong_keyword():
    """Test 'load_pipeline' with a wrong keyword."""
    data = {
        "name": "pipeline-wrong-keyword",
        "pipelineInvalid": [{"transformer": "Distinct"}],
    }
    with pytest.raises(AssertionError):
        load_pipeline(data)


def test_load_pipeline_wrong_nested_keyword():
    """Test 'load_pipeline' with the wrong transformer name."""
    data = {
        "name": "pipeline-wrong-keyword",
        "pipeline": [{"transformer_A": "Distinct"}],
    }
    with pytest.raises(TypeError):
        load_pipeline(data)


def test_load_pipeline_unknown_transformer():
    """Test 'load_pipeline' with an unknown transformer."""
    data = {
        "name": "pipeline-wrong-keyword",
        "pipeline": [{"transformer": "Unknown"}],
    }
    with pytest.raises(NameError):
        load_pipeline(data)


def test_load_pipeline_wrong_transformer_parameters():
    """Test 'load_pipeline' with wrong transformer."""
    data = {
        "name": "pipeline-wrong-keyword",
        "pipeline": [{"transformer": "SelectColumns", "params": {"invalid": None}}],
    }
    with pytest.raises(TypeError):
        load_pipeline(data)


def test_load_pipeline_duplicated_extra_function():
    """Test 'load_pipeline' with duplicated extra-function."""
    data = {"pipeline": [{"transformer": "Count"}]}
    with pytest.raises(AssertionError):
        load_pipeline(data, extra_functions=[my_split_function, my_split_function])


def test_load_pipeline_wrong_data_type():
    """Test 'load_pipeline' with a wrong data type."""
    with pytest.raises(TypeError):
        load_pipeline(1)


def test_load_pipeline_wrong_extra_transformers():
    """Test 'load_pipeline' with wrong extra_transformers."""
    data = {"pipeline": [{"transformer": "Count"}]}
    with pytest.raises(TypeError):
        load_pipeline(data, extra_transformers={1})


def test_invalid_outermost_keys():
    """Test 'load_pipeline' with invalid outermost keys."""
    pipe_cfg = {"invalid": "value", "pipeline": {"transformer": "Distinct"}}
    with pytest.raises(AssertionError):
        load_pipeline(pipe_cfg)


class TestExtractFunction:
    def test_valid(self):
        """Test valid list of functions."""

        def f1(): ...

        def f2(): ...

        chk = _extract_function([f1, f2], "f1")
        assert chk == f1

    def test_invalid(self):
        """Test duplicated list of functions."""

        def f1(): ...

        with pytest.raises(AssertionError):
            _extract_function([f1, f1], "f1")


class TestExtractLazyParams:
    regular_params = {
        "p1": "param_1",
        "p2": 2,
        "p3": [1],
        "p4": {"a": 1, "b": 2},
        "p5": (5,),
    }

    def test_no_lazy_params(self):
        """Tests input with no lazy parameters."""
        chk = extract_lazy_params(self.regular_params)
        assert chk == self.regular_params

    def test_with_ns_param(self):
        """Tests input with a '__ns__' parameter."""
        params = {**{"ns": "__ns__my_key"}, **self.regular_params}
        chk = extract_lazy_params(params)
        chk_lazy = chk.pop("ns")
        assert chk_lazy == (ns, "my_key")
        assert chk == self.regular_params

    def test_with_mixed_params(self):
        """Tests a mix of regular and __ns__ parameters."""
        params = {
            **{"ns": "__ns__my_key"},
            **self.regular_params,
        }
        chk = extract_lazy_params(params)
        chk_lazy_ns = chk.pop("ns")
        assert chk_lazy_ns == (ns, "my_key")
        assert chk == self.regular_params

    def test_empty_input(self):
        """Tests an empty input dictionary."""
        assert extract_lazy_params({}) == {}
