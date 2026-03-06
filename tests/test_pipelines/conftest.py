"""Shared fixtures for pipeline tests."""

import pytest

from nebula.storage import nebula_storage as ns


@pytest.fixture(autouse=True)
def _clear_storage():
    """Clear nebula storage before and after each test."""
    ns.clear()
    yield
    ns.clear()
