"""Transformers for debugging."""

from nlsn.nebula.base import Transformer

__all__ = [
    "Head",
    "Tail",
]


def _check_head_tail(n) -> None:
    if not isinstance(n, int):  # pragma: no cover
        raise ValueError(f'"n" must be <int>, found {n}')

    if n < -1:  # pragma: no cover
        raise ValueError(f'"n" must be >= -1, found {n}')


def _head_tail_func(df, n: int, meth: str):
    if n == -1:
        return df

    return getattr(df, meth)(n)


class Head(Transformer):
    def __init__(self, *, n: int):
        """Return the first 'n' rows of the dataframe.

        Args:
            n (n):
                Limits the result count to the number specified.
                If 0 is provided, it returns an empty dataframe.
                If -1 is provided, it returns the full dataframe.

        Raises:
        - ValueError: If 'n' is not an integer.
        - ValueError: If 'n' is < -1.
        """
        _check_head_tail(n)
        super().__init__()
        self._n: int = int(n)

    def _transform(self, df):
        return _head_tail_func(df, self._n, "head")


class Tail(Transformer):
    def __init__(self, *, n: int):
        """Return the last 'n' rows of the dataframe.

        Args:
            n (n):
                Limits the result count to the number specified.
                If 0 is provided, it returns an empty dataframe.
                If -1 is provided, it returns the full dataframe.

        Raises:
        - ValueError: If 'n' is not an integer.
        - ValueError: If 'n' is < -1.
        """
        _check_head_tail(n)
        super().__init__()
        self._n: int = int(n)

    def _transform(self, df):
        return _head_tail_func(df, self._n, "tail")
