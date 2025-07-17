"""Unit-test for aggregations auxiliary function."""

import pytest

from nlsn.nebula.spark_transformers._constants import (
    ALLOWED_GROUPBY_AGG,
    ALLOWED_WINDOW_AGG,
)
from nlsn.nebula.spark_transformers.aggregations import (
    _Window,
    validate_aggregations,
    validate_window_frame_boundaries,
)


@pytest.mark.parametrize(
    "aggregations",
    [
        {"agg": "collect_list"},
        {"col": "time_bin"},
        {"agg": "collect_list", "alias": "time_bin"},
        {"aggr": "collect_list", "col": "time_bin"},
        {"agg": "sum", "col": "time_bin", "alias": "alias", "wrong": "x"},
    ],
)
@pytest.mark.parametrize("allowed_agg", [ALLOWED_WINDOW_AGG, ALLOWED_GROUPBY_AGG])
@pytest.mark.parametrize(
    "required_keys, allowed_keys, exact_keys",
    [
        [{"agg", "col"}, {"agg", "col", "alias"}, None],
        [None, None, {"agg", "col", "alias"}],
    ],
)
def test_validate_aggregations(
    aggregations, allowed_agg, required_keys, allowed_keys, exact_keys
):
    """Test 'validate_aggregations' auxiliary function."""
    with pytest.raises(ValueError):
        validate_aggregations(
            [aggregations],
            allowed_agg,
            required_keys=required_keys,
            allowed_keys=allowed_keys,
            exact_keys=exact_keys,
        )


@pytest.mark.parametrize(
    "aggregations",
    [
        ({"agg": "sum", "col": 1}),
        ({"agg": "sum", "col": "time_bin", "alias": 1}),
    ],
)
def test_validate_aggregation_types(aggregations):
    """Test 'validate_aggregations' types auxiliary function."""
    with pytest.raises(TypeError):
        validate_aggregations(
            [aggregations],
            ALLOWED_GROUPBY_AGG,
            required_keys={"agg", "col"},
            allowed_keys={"agg", "col", "alias"},
            exact_keys=None,
        )


class TestValidateWindowFrameBoundaries:
    """Unit-test for 'validate_window_frame_boundaries' auxiliary function."""

    def test_valid_window(self):
        """Valid windows."""
        # Test valid integer boundaries
        assert validate_window_frame_boundaries(1, 10) == (1, 10)
        assert validate_window_frame_boundaries(-5, 5) == (-5, 5)
        start, end = validate_window_frame_boundaries("start", "end")
        assert start < 0
        assert end > 5

    def test_invalid_window(self):
        """Invalid windows."""
        # Test None values
        with pytest.raises(ValueError):
            validate_window_frame_boundaries(None, 5)
        with pytest.raises(ValueError):
            validate_window_frame_boundaries(1, None)

        # Test invalid string values
        with pytest.raises(ValueError):
            validate_window_frame_boundaries("invalid", 5)
        with pytest.raises(ValueError):
            validate_window_frame_boundaries(1, "invalid")


class TestWindow:
    """Unit-tests for '_Window' parent class."""

    def test_order_cols_range_between(self):
        """'order_cols' is null when 'range_between' is provided."""
        with pytest.raises(AssertionError):
            _Window(
                partition_cols=["a"],
                order_cols=None,
                ascending=False,
                rows_between=None,
                range_between=(0, 10),
            )

    def test_aggregate_over_window_wrong_ascending(self):
        """Wrong ascending length."""
        with pytest.raises(AssertionError):
            _Window(
                partition_cols=["a"],
                order_cols=["category", "group"],
                ascending=[True, False, False],
                rows_between=None,
                range_between=None,
            )
