"""Datetime utils."""

from string import Template

_PY2J = {
    "Y": "yyyy",
    "m": "MM",
    "d": "dd",
    "H": "HH",
    "M": "mm",
    "S": "ss",
    "f": "SSSSSS",
    "p": "a",
    "A": "EEEE",
    "a": "EEE",
    "Z": "z",
}


def py2java_format(x: str) -> str:
    """Converts a Python datetime format string to a Java datetime format string.

    >>>py2java_format("%Y-%m-%d %H:%M:%S")
    '%Y-%m-%d %H:%M:%S'

    Args:
      x (str): A string representing a Python datetime format.

    Returns (str):
      A string representing the equivalent Java datetime format.
    """
    return Template(x.replace("%", "$")).substitute(**_PY2J)
