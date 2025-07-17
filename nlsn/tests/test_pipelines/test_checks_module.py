"""Unit-testing for pipeline checks.py module."""

# pylint: disable=unused-wildcard-import
# pylint: disable=undefined-variable
# pylint: disable=unused-wildcard-import

from typing import Optional

import pytest

from nlsn.nebula.pipelines._checks import *


@pytest.mark.parametrize(
    "o",
    [
        {"input_col": "x", "operator": "eq"},
        {"input_col": "x", "operator": "eq", "skip_if_empty": True},
        {"input_col": "x", "operator": "eq", "value": None},
        {"input_col": "x", "operator": "isNotNull", "dead-end": True},
        {"input_col": "x", "operator": "eq", "comparison_column": "y"},
    ],
)
def test_apply_to_rows_valid_configuration(o):
    """Valid 'apply_to_rows' configuration."""
    assert_apply_to_rows_inputs(o)


@pytest.mark.parametrize(
    "o",
    [
        {"input_col": "x", "operator": "eq", "value": 0, "comparison_column": "y"},
        {"input_col": "x", "operator": "eq", "comparison_column": "x"},
        {"input_col": "x", "operator": "eq", "skip_if_empty": "wrong"},
    ],
)
def test_apply_to_rows_wrong_configuration(o):
    """Wrong 'apply_to_rows' configuration."""
    with pytest.raises(ValueError):
        assert_apply_to_rows_inputs(o)


def test_apply_to_rows_not_allowed_key():
    """Wrong 'apply_to_rows' keys."""
    with pytest.raises(KeyError):
        assert_apply_to_rows_inputs({"input_col": "x", "wrong": "x"})


@pytest.mark.parametrize(
    "o",
    [
        {"end": "dead-end"},
        {"end": "join", "on": "key1", "how": "inner"},
        {"end": "append", "storage": "x"},
    ],
)
@pytest.mark.parametrize("add_storage", [False, True])
def test_branch_valid_configuration(o, add_storage: bool):
    """Valid 'branch' configuration."""
    if add_storage:
        input_o = o.copy()
        input_o["storage"] = "x"
    else:
        input_o = o
    assert_branch_inputs(input_o)


def test_branch_not_allowed_key():
    """Wrong 'branch' configuration."""
    with pytest.raises(KeyError):
        assert_branch_inputs({"end": "dead-end", "wrong": "x"})


def test_branch_empty():
    """Empty 'branch' configuration."""
    with pytest.raises(KeyError):
        assert_branch_inputs({})


def test_branch_invalid_type_key():
    """Wrong 'end' key for 'branch' configuration."""
    with pytest.raises(ValueError):
        assert_branch_inputs({"end": "invalid_type"})


@pytest.mark.parametrize("o", ["str", []])
def test_branch_wrong_argument_type(o):
    """Wrong input format for 'branch' configuration."""
    with pytest.raises(TypeError):
        assert_branch_inputs(o)


def test_branch_extra_keys_for_dead_end():
    """Not allowed keys for dead-end 'branch' configuration."""
    invalid_input = {"end": "dead-end", "extra_key": "value"}
    with pytest.raises(KeyError):
        assert_branch_inputs(invalid_input)


@pytest.mark.parametrize(
    "o",
    [
        {"end": "append", "extra_key": "value"},
        {"end": "append", "on": "x"},
        {"end": "append", "how": "x"},
        {"end": "dead-end", "extra_key": "value"},
        {"end": "dead-end", "on": "x"},
        {"end": "dead-end", "how": "x"},
    ],
)
def test_branch_extra_keys_for_append(o):
    """Not allowed keys for append / dead-end 'branch' configuration."""
    with pytest.raises(KeyError):
        assert_branch_inputs(o)


def test_branch_extra_keys_for_join():
    """Extra join keys for join 'branch' configuration."""
    invalid_input = {"end": "join", "on": "1", "how": "inner", "extra": "x"}
    with pytest.raises(KeyError):
        assert_branch_inputs(invalid_input)


def test_branch_missing_keys_for_join():
    """Missing join keys for join 'branch' configuration."""
    invalid_input = {"end": "join", "on": "x"}
    with pytest.raises(KeyError):
        assert_branch_inputs(invalid_input)


class TestEnsureNoBranchOrApplyToRowsOtherwise:
    @pytest.mark.parametrize(
        "branch, apply_to_rows, otherwise",
        [
            ({"end": "some-end"}, None, None),
            (None, {"dead-end": False}, None),
            ({"end": "some-end"}, None, {"some-key": "value"}),
            (None, {"dead-end": False}, {"some-key": "value"}),
        ],
    )
    def test_valid_cases(self, branch, apply_to_rows, otherwise):
        """Test valid cases."""
        ensure_no_branch_or_apply_to_rows_otherwise(branch, apply_to_rows, otherwise)

    @pytest.mark.parametrize(
        "branch, apply_to_rows, otherwise",
        [
            ({"storage": "x"}, None, {"some-key": "value"}),
            ({"end": "dead-end"}, None, {"some-key": "value"}),
            (None, {"dead-end": True}, {"some-key": "value"}),
            (None, None, {"some-key": "value"}),
            ({"end": "some-end"}, {"dead-end": True}, None),
        ],
    )
    def test_invalid_cases(self, branch, apply_to_rows, otherwise):
        """Test invalid cases."""
        with pytest.raises(AssertionError):
            ensure_no_branch_or_apply_to_rows_otherwise(
                branch, apply_to_rows, otherwise
            )


def test_ensure_no_branch_or_apply_to_rows_in_split_pipeline():
    """Test ensure_no_branch_or_apply_to_rows_in_split_pipeline function."""
    with pytest.raises(AssertionError):
        ensure_no_branch_or_apply_to_rows_in_split_pipeline({"end": "some-end"}, None)

    with pytest.raises(AssertionError):
        ensure_no_branch_or_apply_to_rows_in_split_pipeline(
            None, {"input_col": "x", "operator": "eq"}
        )

    # This should not raise an exception
    ensure_no_branch_or_apply_to_rows_in_split_pipeline(None, None)


class TestSkipPerformValidation:
    """Test 'validate_skip_perform'."""

    @staticmethod
    @pytest.mark.parametrize("skip, perform", [[True, True], [False, False]])
    def test_invalid(skip, perform):
        """Should raise AssertionError."""
        with pytest.raises(AssertionError):
            validate_skip_perform(skip, perform)

    @staticmethod
    @pytest.mark.parametrize(
        "skip, perform, expected",
        [
            [True, False, True],  # skipped
            [False, True, False],
            [True, None, True],  # skipped
            [False, None, False],
            [None, True, False],
            [None, False, True],  # skipped
            [None, None, False],
        ],
    )
    def test_valid(skip: Optional[bool], perform: Optional[bool], expected: bool):
        """Assert expected combinations."""
        assert validate_skip_perform(skip, perform) is expected
