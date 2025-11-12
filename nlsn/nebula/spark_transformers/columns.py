"""Transformers for Managing Columns without Affecting Row Values."""

import pyspark.sql.functions as F

from nlsn.nebula.auxiliaries import (
    assert_is_string,
    assert_only_one_non_none,
)
from nlsn.nebula.base import Transformer

__all__ = [
    "DuplicateColumn",
]


class DuplicateColumn(Transformer):
    def __init__(
            self,
            *,
            col_map: dict[str, str] | None = None,
            pairs: list[tuple[str, str]] | None = None,
    ):
        """Duplicate columns and assign them new names.

        If a new duplicate column has an existing name, the old column will
        be dropped.

        Only one among col_map and pairs can be provided.

        Args:
            col_map (dict(str, str) | None):
                A dictionary specifying column duplications. Keys represent
                the original column names, and values represent the desired
                new column names. Note that due to the unordered nature
                of dictionaries, the order of duplications is not guaranteed.
                For example, if you have `{"x": "x_dup", "y": "x"}` and the
                "y" mapping is processed first, "x_dup" will be a duplicate of
                "y", not the original "x". To ensure a specific order of
                duplications, use the `pairs` argument instead.
                Defaults to None.
            pairs (list(iterable(str, str)) | None):
                A list of 2-element iterable, where each tuple represents a
                column duplication. The first element of each tuple is the
                original column name, and the second element is the new
                column name. Duplications are performed in the order they
                appear in the list, providing explicit control over the
                sequence of operations. This addresses potential ordering
                issues that can arise when using `col_map`.
                Defaults to None.
        """
        assert_only_one_non_none(col_map, pairs)

        super().__init__()

        self._pairs: list[tuple[str, str]] = []
        if pairs:
            if not all(len(i) == 2 for i in pairs):
                msg = "'pairs' must be a list of 2-element iterables. Found different lengths"
                raise AssertionError(msg)
            self._pairs = pairs
        else:
            set_keys: set[str] = set(col_map.keys())
            values = col_map.values()
            ints = set_keys.intersection(set(values))
            if ints:
                msg = (
                    f"Keys and values must be all different. Found intersection {ints}"
                )
                raise AssertionError(msg)

            self._pairs = list(col_map.items())

        values = [i[1] for i in self._pairs]
        set_values: set[str] = set(values)
        if len(values) != len(set_values):
            raise AssertionError(f"Duplicated values in mapping: {values}")

    def _transform(self, df):
        new_column_names: set[str] = {i[1] for i in self._pairs}
        old_cols = [i for i in df.columns if i not in new_column_names]
        new_cols = [F.col(k).alias(v) for k, v in self._pairs]
        return df.select(*old_cols, *new_cols)
