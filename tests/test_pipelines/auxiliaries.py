"""Auxiliaries for testing pipelines."""

import os
from dataclasses import dataclass
from pathlib import Path

import narwhals as nw
import yaml

from nebula.base import Transformer
from nebula.storage import nebula_storage as ns

__all__ = [
    "load_yaml",
    "AddOne",
    "CallMe",
    "Distinct",
    "NoParentClass",
    "RoundValues",
    "ThisTransformerIsBroken",
    "ExtraTransformers",  # DataClass
]

_this_path = Path(os.path.dirname(os.path.realpath(__file__)))


class AddOne(Transformer):

    def __init__(self, column: str):
        super().__init__()
        self._col: str = column

    def _transform_nw(self, df):
        if self._col in df.columns:
            value = df[self._col].max() + 1
        else:
            value = 0
        df = df.with_columns(nw.lit(value).alias(self._col))
        return df


class CallMe(Transformer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _transform_nw(df):
        if "_call_me_" in ns.list_keys():
            value = ns.get("_call_me_") + 1
            ns.set("_call_me_", value)
        else:
            ns.set("_call_me_", 1)
        return df


class Distinct(Transformer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _transform_nw(df):
        return df.unique()


class NoParentClass:
    @staticmethod
    def transform(df):
        return df


class RoundValues(Transformer):

    def __init__(self, *, column: str, precision: int):
        super().__init__()
        self._col = column
        self._precision = precision

    def _transform_nw(self, df):
        return df.with_columns(
            nw.col(self._col).round(self._precision).alias(self._col)
        )


class ThisTransformerIsBroken(Transformer):
    @staticmethod
    def _transform_nw(df):
        raise ValueError("Broken transformer")


@dataclass
class ExtraTransformers:
    AddOne = AddOne
    CallMe = CallMe
    Distinct = Distinct
    RoundValues = RoundValues
    ThisTransformerIsBroken = ThisTransformerIsBroken


def load_yaml(path) -> dict:
    """Load a YAML file to test the 'load_pipeline' function."""
    with open(_this_path / "yml_files" / path, "r", encoding="utf-8") as stream:
        ret = yaml.safe_load(stream)
    return ret
