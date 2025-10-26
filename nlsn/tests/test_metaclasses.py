"""Unit-tests for meta-classes."""

import pytest

from nlsn.nebula.metaclasses import InitParamsStorage


class Parent(metaclass=InitParamsStorage):
    def __init__(self):
        """Parent class."""
        self._transformer_init_params: dict = {}

    @property
    def transformer_init_parameters(self) -> dict:
        """Set the initialization parameters."""
        return self._transformer_init_params


class Child(Parent):
    def __init__(self, *, x, y=None):
        """Child class."""
        super().__init__()
        self.x = x
        self.y = y


def test_init_params_storage():
    """Unit-test for 'InitParamsStorage' metaclass."""
    params_one_exp = {"x": 10}
    params_two_exp = {"x": 20, "y": -5}

    child_one = Child(**params_one_exp)
    child_two = Child(**params_two_exp)

    # Test 'child_two' first, and then proceed to test 'child_one'.
    params_two_chk = child_two.transformer_init_parameters
    assert params_two_chk == params_two_exp

    params_one_chk = child_one.transformer_init_parameters
    assert params_one_chk == params_one_exp


def test_init_params_storage_error():
    """Unit-test for 'InitParamsStorage' with wrong class initialization."""
    with pytest.raises(TypeError):
        Child(x="x", c=5)
