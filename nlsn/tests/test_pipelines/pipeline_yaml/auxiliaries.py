"""Auxiliaries functions for pipeline loader unit-tests."""

import os
from pathlib import Path

import yaml

__all__ = ["load_yaml"]

this_path = Path(os.path.dirname(os.path.realpath(__file__)))


def load_yaml(path) -> dict:
    """Load a YAML file to test the 'load_pipeline' function."""
    with open(this_path / path, "r", encoding="utf-8") as stream:
        ret = yaml.safe_load(stream)
    return ret
