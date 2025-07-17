"""Unit-tests for deprecation function."""

import pytest

from nlsn.nebula.deprecations import deprecate_transformer
from nlsn.nebula.shared_transformers import RoundValues


@pytest.mark.parametrize("msg", ["RoundDecimalValues", ""])
def test_deprecate_transformer(msg: str):
    """Test 'deprecate_transformer' function."""
    deprecate_transformer(RoundValues, msg)
