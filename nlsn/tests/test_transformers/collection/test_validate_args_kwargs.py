"""Unittests for 'validate_args_kwargs' function in spark_transformers."""

import pytest

from nlsn.nebula.spark_transformers.collection import validate_args_kwargs


def test_validate_args_kwargs_wrong_args():
    """Test validate_args_kwargs with wrong args."""
    with pytest.raises(TypeError):
        validate_args_kwargs("invalid_args")


@pytest.mark.parametrize("kws", ["invalid_kws", {"x": "v1", 2: "v2"}])
def test_validate_args_kwargs_wrong_kwargs(kws):
    """Test validate_args_kwargs with wrong kwargs."""
    with pytest.raises(TypeError):
        validate_args_kwargs(None, kws)
