"""Unit-test for CompareStoredValues."""

import pytest

from nlsn.nebula.spark_transformers import CompareStoredValues
from nlsn.nebula.storage import nebula_storage as ns


@pytest.mark.parametrize(
    "kwargs, store_b, error",
    [
        ({"value": 1, "comparison": "lt"}, None, True),
        ({"value": 1, "comparison": "gt"}, None, False),
        ({"value": 10, "comparison": "eq"}, None, False),
        ({"value": [1, 10], "comparison": "isin"}, None, False),
        ({"value": [1, 11], "comparison": "isin"}, None, True),
        ({"key_b": "right_value", "comparison": "lt"}, 1, True),
        ({"key_b": "right_value", "comparison": "gt"}, 1, False),
        ({"key_b": "right_value", "comparison": "eq"}, 10, False),
        ({"key_b": "right_value", "comparison": "isin"}, [1, 10], False),
        ({"key_b": "right_value", "comparison": "isin"}, [1, 11], True),
    ],
)
def test_compare_stored_values(kwargs, store_b: int, error: bool):
    """Test CompareStoredValues transformer."""
    ns.clear()
    ns.set("reference", 10)

    if "key_b" in kwargs:
        ns.set(kwargs["key_b"], store_b)

    t = CompareStoredValues(key_a="reference", **kwargs)

    if error:
        with pytest.raises(AssertionError):
            t.transform("df")
    else:
        exp = t.transform("df")
        # No operations on the dataframe are performed.
        assert exp == "df"
