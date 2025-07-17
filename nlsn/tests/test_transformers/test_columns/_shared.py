"""Shared util for 'columns' transformers."""

from itertools import product
from random import randint, sample

import numpy as np

_columns_selector = [f"{a}{b}" for a, b in product(["c", "d"], range(5))]


def _generate_data_column_selector():
    n_rows = 10
    shape = (n_rows, len(_columns_selector))
    return np.random.randint(0, 100, shape).tolist()


class SharedDropColumns:
    columns = _columns_selector

    columns_params = "columns", [
        None,
        *(sample(_columns_selector, randint(0, 5)) for _ in range(3)),
    ]
    regex_params = "regex", [None, "c1", "c[1-5]", "c[46]"]
    glob_params = "glob", [None, "", "c*"]

    @staticmethod
    def generate_data():
        """Data for DropColumns."""
        return _generate_data_column_selector()


class SharedRenameColumns:
    data = [
        ["a", 0.0],
        ["a1", 1.0],
        ["a", 2.0],
        ["a1", 3.0],
        ["b1", 4.0],
    ]
    columns = ["c1", "c2"]

    set_str = {i[0] for i in data}
    set_flt = {i[1] for i in data}

    params = (
        "mapping, columns, columns_renamed, regex_pattern, regex_replacement, exp",
        [
            ({"c2": "c2a", "c1": "c1a"}, None, None, None, None, ["c1a", "c2a"]),
            (None, ["c2", "c1"], ["c2a", "c1a"], None, None, ["c1a", "c2a"]),
            (None, None, None, "^c1", "c1_new", ["c1_new", "c2"]),
        ],
    )


class SharedSelectColumns:
    columns = _columns_selector

    columns_params = "columns", [
        None,
        *(sample(_columns_selector, randint(1, 8)) for _ in range(3)),
    ]
    regex_params = "regex", [None, "a^", "^c", "^d", "[c-d]1"]
    glob_params = "glob", [None, "*", "", "c*"]

    @staticmethod
    def generate_data():
        """Data for DropColumns."""
        return _generate_data_column_selector()
