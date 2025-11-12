"""Transformers for debugging."""

from nlsn.nebula.auxiliaries import assert_cmp, assert_is_integer
from nlsn.nebula.base import Transformer

__all__ = [
    "Limit",
]


class Limit(Transformer):
    def __init__(self, *, n: int):
        """Limit the result count to the number specified.

        Args:
            n (n):
                Limits the result count to the number specified.
                If 0 is provided, it returns an empty dataframe.
                If -1 is provided, it returns the full dataframe.

        Raises:
        - ValueError: If 'n' is not an integer.
        - ValueError: If 'n' is < -1.
        """
        assert_is_integer(n, "n")
        assert_cmp(n, "ge", -1, "n")

        super().__init__()
        self._n: int = int(n)

    def _transform(self, df):
        if self._n == -1:
            return df

        return df.limit(self._n)
