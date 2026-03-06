"""Unit tests for function-dict support in pipeline_loader.

Tests cover _load_function() directly and its integration through
load_pipeline() / _load_generic().

Dict shape under test:
    {
        "function": <callable | str>,
        "args":     [...],       # optional
        "kwargs":   {...},       # optional
        "description": "...",   # optional
        "skip":     True,        # optional
        "perform":  False,       # optional alternative skip
    }
"""

import polars as pl
import pytest

from nebula.pipelines.pipeline_loader import _load_function, load_pipeline

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _identity(df):
    """Return the DataFrame unchanged."""
    return df


def _add_col(df, col_name="extra", value=0):
    """Add a literal column to the DataFrame."""
    return df.with_columns(pl.lit(value).alias(col_name))


def _multiply_col(df, col, factor=2):
    """Multiply an existing column by a scalar factor."""
    return df.with_columns((pl.col(col) * factor).alias(col))


extra_functions = {
    "identity": _identity,
    "add_col": _add_col,
    "multiply_col": _multiply_col,
}


@pytest.fixture
def sample_df():
    return pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})


# ---------------------------------------------------------------------------
# _load_function – unit tests
# ---------------------------------------------------------------------------


class TestLoadFunctionReturnValues:
    """_load_function returns the correct tuple shape."""

    def test_callable_no_args_returns_callable(self):
        result = _load_function({"function": _identity}, extra_funcs={})
        assert result is _identity

    def test_callable_with_args_returns_two_tuple(self):
        result = _load_function({"function": _add_col, "args": ["my_col"]}, extra_funcs={})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is _add_col
        assert result[1] == ["my_col"]

    def test_callable_with_kwargs_returns_three_tuple(self):
        result = _load_function(
            {"function": _add_col, "kwargs": {"col_name": "x", "value": 99}},
            extra_funcs={},
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        func, args, kwargs = result
        assert func is _add_col
        assert args == []
        assert kwargs == {"col_name": "x", "value": 99}

    def test_callable_with_description_returns_four_tuple(self):
        result = _load_function(
            {"function": _identity, "description": "my step"},
            extra_funcs={},
        )
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert result[0] is _identity
        assert result[1] == []
        assert result[2] == {}
        assert result[3] == "my step"

    def test_callable_with_args_kwargs_description(self):
        result = _load_function(
            {
                "function": _add_col,
                "args": ["new_col"],
                "kwargs": {"value": 7},
                "description": "adds new_col",
            },
            extra_funcs={},
        )
        func, args, kwargs, desc = result
        assert func is _add_col
        assert args == ["new_col"]
        assert kwargs == {"value": 7}
        assert desc == "adds new_col"


class TestLoadFunctionStringLookup:
    """_load_function resolves function names from extra_funcs."""

    def test_string_name_resolved(self):
        result = _load_function({"function": "identity"}, extra_funcs=extra_functions)
        assert result is _identity

    def test_string_name_with_kwargs(self):
        result = _load_function(
            {"function": "add_col", "kwargs": {"col_name": "z", "value": 5}},
            extra_funcs=extra_functions,
        )
        func, args, kwargs = result
        assert func is _add_col
        assert kwargs == {"col_name": "z", "value": 5}

    def test_unknown_string_name_raises_name_error(self):
        with pytest.raises(NameError, match="Unknown function"):
            _load_function({"function": "nonexistent"}, extra_funcs=extra_functions)

    def test_unknown_string_empty_registry_raises_name_error(self):
        with pytest.raises(NameError, match="Unknown function"):
            _load_function({"function": "anything"}, extra_funcs={})


class TestLoadFunctionSkip:
    """_load_function respects skip / perform flags."""

    def test_skip_true_returns_none(self):
        result = _load_function({"function": "identity", "skip": True}, extra_funcs=extra_functions)
        assert result is None

    def test_perform_false_returns_none(self):
        result = _load_function({"function": "identity", "perform": False}, extra_funcs=extra_functions)
        assert result is None

    def test_skip_false_not_skipped(self):
        result = _load_function({"function": "identity", "skip": False}, extra_funcs=extra_functions)
        assert result is _identity

    def test_perform_true_not_skipped(self):
        result = _load_function({"function": "identity", "perform": True}, extra_funcs=extra_functions)
        assert result is _identity


class TestLoadFunctionValidation:
    """_load_function validates bad inputs."""

    def test_non_callable_non_string_raises_type_error(self):
        with pytest.raises(TypeError, match='"function" must be a callable or a string name'):
            _load_function({"function": 42}, extra_funcs={})

    def test_args_not_list_raises_type_error(self):
        with pytest.raises(TypeError, match='"args" must be a list or tuple'):
            _load_function({"function": _identity, "args": "not_a_list"}, extra_funcs={})

    def test_kwargs_not_dict_raises_type_error(self):
        with pytest.raises(TypeError, match='"kwargs" must be a dict'):
            _load_function({"function": _identity, "kwargs": ["not", "a", "dict"]}, extra_funcs={})


# ---------------------------------------------------------------------------
# load_pipeline integration tests
# ---------------------------------------------------------------------------


class TestLoadPipelineWithFunctionDict:
    """Function dicts work correctly through the full load_pipeline path."""

    def test_plain_callable_in_pipeline(self, sample_df):
        pipe = load_pipeline(
            [{"function": _identity}],
            extra_functions=extra_functions,
        )
        result = pipe.run(sample_df)
        assert result.equals(sample_df)

    def test_string_function_name_in_pipeline(self, sample_df):
        pipe = load_pipeline(
            [{"function": "identity"}],
            extra_functions=extra_functions,
        )
        result = pipe.run(sample_df)
        assert result.equals(sample_df)

    def test_function_with_kwargs_executes_correctly(self, sample_df):
        pipe = load_pipeline(
            [{"function": "add_col", "kwargs": {"col_name": "extra", "value": 42}}],
            extra_functions=extra_functions,
        )
        result = pipe.run(sample_df)
        assert "extra" in result.columns
        assert result["extra"].to_list() == [42, 42, 42]

    def test_function_with_args_executes_correctly(self, sample_df):
        pipe = load_pipeline(
            [{"function": "add_col", "args": ["my_new_col"]}],
            extra_functions=extra_functions,
        )
        result = pipe.run(sample_df)
        assert "my_new_col" in result.columns

    def test_function_mixed_with_transformers(self, sample_df):
        pipe = load_pipeline(
            [
                {"transformer": "SelectColumns", "params": {"columns": ["a", "b"]}},
                {"function": "multiply_col", "kwargs": {"col": "a", "factor": 10}},
            ],
            extra_functions=extra_functions,
        )
        result = pipe.run(sample_df)
        assert result["a"].to_list() == [10, 20, 30]

    def test_function_skipped_via_skip_flag(self, sample_df):
        """A skipped function dict is silently dropped – df passes through unchanged."""
        pipe = load_pipeline(
            [
                {"function": "add_col", "kwargs": {"col_name": "should_not_exist"}, "skip": True},
            ],
            extra_functions=extra_functions,
        )
        result = pipe.run(sample_df)
        assert "should_not_exist" not in result.columns

    def test_function_skipped_via_perform_false(self, sample_df):
        pipe = load_pipeline(
            [
                {"function": "add_col", "kwargs": {"col_name": "should_not_exist"}, "perform": False},
            ],
            extra_functions=extra_functions,
        )
        result = pipe.run(sample_df)
        assert "should_not_exist" not in result.columns

    def test_multiple_function_dicts_in_sequence(self, sample_df):
        pipe = load_pipeline(
            [
                {"function": "add_col", "kwargs": {"col_name": "x", "value": 1}},
                {"function": "add_col", "kwargs": {"col_name": "y", "value": 2}},
            ],
            extra_functions=extra_functions,
        )
        result = pipe.run(sample_df)
        assert "x" in result.columns
        assert "y" in result.columns

    def test_unknown_function_name_raises_name_error(self, sample_df):
        with pytest.raises(NameError, match="Unknown function"):
            load_pipeline(
                [{"function": "does_not_exist"}],
                extra_functions=extra_functions,
            )

    def test_function_dict_with_description_runs_correctly(self, sample_df):
        """Description is metadata only – pipeline still executes normally."""
        pipe = load_pipeline(
            [{"function": "identity", "description": "passthrough step"}],
            extra_functions=extra_functions,
        )
        result = pipe.run(sample_df)
        assert result.equals(sample_df)

    def test_function_before_and_after_transformer(self, sample_df):
        pipe = load_pipeline(
            [
                {"function": "multiply_col", "kwargs": {"col": "a", "factor": 2}},
                {"transformer": "SelectColumns", "params": {"columns": ["a"]}},
                {"function": "add_col", "kwargs": {"col_name": "tag", "value": 99}},
            ],
            extra_functions=extra_functions,
        )
        result = pipe.run(sample_df)
        assert list(result.columns) == ["a", "tag"]
        assert result["a"].to_list() == [2, 4, 6]
        assert result["tag"].to_list() == [99, 99, 99]
