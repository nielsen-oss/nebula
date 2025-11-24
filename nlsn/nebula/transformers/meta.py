"""Row Filtering Operations."""

from typing import Any

import narwhals as nw

from nlsn.nebula.auxiliaries import assert_allowed
from nlsn.nebula.base import Transformer

__all__ = [
    "DataFrameMethod",
]

_NW_DATAFRAME_METHODS: set[str] = {
    i for i in dir(nw.DataFrame) if i.islower() and i[0] != "_"
}
_NW_LAZYFRAME_METHODS: set[str] = {
    i for i in dir(nw.LazyFrame) if i.islower() and i[0] != "_"
}
_NW_METHODS: set[str] = _NW_DATAFRAME_METHODS.union(_NW_LAZYFRAME_METHODS)


class DataFrameMethod(Transformer):
    """Call a Narwhals DataFrame or LazyFrame method.

    Meta-transformer that provides direct access to the Narwhals API,
    allowing you to call any DataFrame or LazyFrame method from your
    pipeline configuration.

    Examples:
        >>> # Drop duplicates
        >>> t = DataFrameMethod(method="unique")
        >>> df_out = t.transform(df)

        >>> # Sample with seed
        >>> t = DataFrameMethod(
        ...     method="sample",
        ...     kwargs={"n": 1000, "seed": 42}
        ... )
        >>> df_out = t.transform(df)

        >>> # Sort by columns
        >>> t = DataFrameMethod(
        ...     method="sort",
        ...     args=[["user_id", "date"]],
        ...     kwargs={"descending": [False, True]}
        ... )
        >>> df_out = t.transform(df)
    """

    def __init__(
            self,
            *,
            method: str,
            args: list[Any] | None = None,
            kwargs: dict[str, Any] | None = None,
    ):
        """Initialize DataFrameMethod transformer.

        Args:
            method: Name of the Narwhals method to call.
            args: Positional arguments to pass to the method. Defaults to None.
            kwargs: Keyword arguments to pass to the method. Defaults to None.

        Raises:
            ValueError: If method is not a valid Narwhals method.
        """
        assert_allowed(method, _NW_METHODS, "method")

        super().__init__()
        self._method_name: str = method
        self._args: list = args if args else []
        self._kwargs: dict[str, Any] = kwargs if kwargs else {}

    def _transform_nw(self, df):
        """Call the specified method on the Narwhals DataFrame or LazyFrame.

        Args:
            df: Narwhals DataFrame or LazyFrame.

        Returns:
            Result of calling the method (typically a DataFrame/LazyFrame).

        Raises:
            AttributeError: If method is not available for the frame type.
        """
        # Validate method availability for the specific frame type
        if isinstance(df, nw.LazyFrame):
            if self._method_name not in _NW_LAZYFRAME_METHODS:
                msg = (
                    f"Method '{self._method_name}' is not available for LazyFrame. "
                    f"This method is only available on eager DataFrames. "
                    f"Available LazyFrame methods: {sorted(_NW_LAZYFRAME_METHODS)}"
                )
                raise AttributeError(msg)
        else:  # nw.DataFrame
            if self._method_name not in _NW_DATAFRAME_METHODS:
                msg = (
                    f"Method '{self._method_name}' is not available for DataFrame. "
                    f"This method is only available on LazyFrames. "
                    f"Available DataFrame methods: {sorted(_NW_DATAFRAME_METHODS)}"
                )
                raise AttributeError(msg)

        # Call the method with provided args and kwargs
        method = getattr(df, self._method_name)
        return method(*self._args, **self._kwargs)
