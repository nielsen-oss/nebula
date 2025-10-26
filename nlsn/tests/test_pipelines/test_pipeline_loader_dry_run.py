"""Test the pipeline loader starting from YAML data with a dry-run."""

import pytest

from nlsn.nebula.pipelines.pipeline_loader import (
    _extract_function,
    extract_lazy_params,
    load_pipeline,
)
from nlsn.nebula.storage import nebula_storage as ns
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml


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
    fname = "wrong_splits.yml"
    data = load_yaml(fname)[pipeline_key]

    with pytest.raises(AssertionError):
        load_pipeline(data, extra_functions=exposed_functions)


@pytest.mark.parametrize("pipeline_key", ["pipeline_1"])
def test_load_pipeline_mock_dry_run(pipeline_key: str):
    """Test wrong split-pipelines."""
    fname = "mock.yml"
    data = load_yaml(fname)[pipeline_key]

    pipe = load_pipeline(data, extra_functions=exposed_functions)

    pipe.show_pipeline(add_transformer_params=True)
    pipe._print_dag()


def test_load_pipeline_wrong_keyword():
    """Test 'load_pipeline' with a wrong keyword."""
    data = {
        "name": "pipeline-wrong-keyword",
        "pipelineInvalid": [{"transformer": "Distinct"}],
    }
    with pytest.raises(AssertionError):
        load_pipeline(data)


def test_load_pipeline_wrong_nested_keyword():
    """Test 'load_pipeline' with a wrong transformer name."""
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
        "pipeline": [{"transformer": "Count", "params": {"wrong": None}}],
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

        def f1():
            ...

        def f2():
            ...

        chk = _extract_function([f1, f2], "f1")
        assert chk == f1

    def test_invalid(self):
        """Test duplicated list of functions."""

        def f1():
            ...

        with pytest.raises(AssertionError):
            _extract_function([f1, f1], "f1")


class TestExtractLazyParams:
    extra_funcs = {"func": lambda: "lambda"}
    regular_params = {"p1": "param_1", "p2": 2, "p3": [1], "p4": {"a": 1, "b": 2}}

    def test_no_lazy_params(self):
        """Tests input with no lazy parameters."""
        chk = extract_lazy_params(self.regular_params, self.extra_funcs)
        assert chk == self.regular_params

    def test_with_fn_param(self):
        """Tests input with a '__fn__' parameter."""
        params = {**{"func": "__fn__func"}, **self.regular_params}
        chk = extract_lazy_params(params, self.extra_funcs)
        chk_lazy = chk.pop("func")
        assert chk_lazy is self.extra_funcs["func"]
        assert chk == self.regular_params

    def test_with_ns_param(self):
        """Tests input with a '__ns__' parameter."""
        params = {**{"ns": "__ns__my_key"}, **self.regular_params}
        chk = extract_lazy_params(params, self.extra_funcs)
        chk_lazy = chk.pop("ns")
        assert chk_lazy == (ns, "my_key")
        assert chk == self.regular_params

    def test_with_mixed_params(self):
        """Tests a mix of regular, __fn__, and __ns__ parameters."""
        params = {
            **{"func": "__fn__func"},
            **{"ns": "__ns__my_key"},
            **self.regular_params,
        }
        chk = extract_lazy_params(params, self.extra_funcs)
        chk_lazy_ns = chk.pop("ns")
        chk_lazy_func = chk.pop("func")
        assert chk_lazy_ns == (ns, "my_key")
        assert chk_lazy_func is self.extra_funcs["func"]
        assert chk == self.regular_params

    def test_empty_input(self):
        """Tests an empty input dictionary."""
        assert extract_lazy_params({}, self.extra_funcs) == {}

    def test_fn_not_found(self):
        """Tests that a KeyError is raised for a non-existent function."""
        params = {"func": "__fn__non_existent_func"}
        with pytest.raises(KeyError):
            extract_lazy_params(params, self.extra_funcs)
