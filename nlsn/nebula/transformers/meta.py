"""Row Filtering Operations."""

from typing import Any, Iterable

import narwhals as nw

from nlsn.nebula.auxiliaries import assert_allowed
from nlsn.nebula.base import Transformer

__all__ = [
    "DataFrameMethod",
    "WithColumns",
]


def _get_public_methods(cls) -> set[str]:
    return {i for i in dir(cls) if i.islower() and i[0] != "_"}


_NW_DATAFRAME_METHODS: set[str] = _get_public_methods(nw.DataFrame)
_NW_LAZYFRAME_METHODS: set[str] = _get_public_methods(nw.LazyFrame)
_NW_DF_METHODS: set[str] = _NW_DATAFRAME_METHODS.union(_NW_LAZYFRAME_METHODS)

_NW_FLAT_COL_METHODS: set[str] = _get_public_methods(nw.col())
_NW_NESTED_COL_METHODS: dict[str, set[str]] = {
    "str": _get_public_methods(nw.col().str),
    "dt": _get_public_methods(nw.col().dt),
}


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
        assert_allowed(method, _NW_DF_METHODS, "dataframe-method")

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


class WithColumns(Transformer):
    def __init__(
            self,
            *,
            columns: str | list[str] | None = None,
            regex: str | None = None,
            glob: str | None = None,
            startswith: str | Iterable[str] | None = None,
            endswith: str | Iterable[str] | None = None,
            method: str,
            args: list[Any] | None = None,
            kwargs: dict[str, Any] | None = None,
            prefix: str | None = None,
            suffix: str | None = None,
    ):
        """Apply a Narwhals method to multiple columns.

        Args:
            columns (str | list(str) | None):
                A list of columns to select. Defaults to None.
            regex (str | None):
                Take the columns to select by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Take the columns to select by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
            method (str):
                Method name (e.g., 'round', 'str.strip_chars', 'dt.year').
            args:
                Positional arguments for the method.
            kwargs:
                Keyword arguments for the method.
            prefix (str | None):
                Add prefix to output column names.
            suffix (str | None):
                Add suffix to output column names.
        """
        super().__init__()
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )
        method_splits = method.split(".")
        n_split = len(method_splits)
        self._method_accessor: str | None = None
        self._method_name: str
        if n_split == 1:
            assert_allowed(method, _NW_FLAT_COL_METHODS, "column-method")
            self._method_accessor = None
            self._method_name = method_splits[0]
        elif n_split == 2:
            accessor: str = method_splits[0]
            assert_allowed(accessor, {"str", "dt"}, "column-accessor")
            allowed = _NW_NESTED_COL_METHODS[accessor]
            assert_allowed(method_splits[1], allowed, "column-method")
            self._method_accessor = accessor
            self._method_name = method_splits[1]
        else:
            raise ValueError(
                f"Method '{method}' has too many parts. "
                f"Expected format: 'method' or 'namespace.method'"
            )

        self._args: list = args if args else []
        self._kwargs: dict[str, Any] = kwargs if kwargs else {}
        self._prefix: str | None = prefix
        self._suffix: str | None = suffix

    def _transform_nw(self, df):
        selection: list[str] = self._get_selected_columns(df)

        if not selection:  # Pass through if no columns selected
            return df

        meth = nw.col(*selection)

        if self._method_accessor:
            meth = getattr(meth, self._method_accessor)
        meth = getattr(meth, self._method_name)

        func = meth(*self._args, **self._kwargs)

        if self._prefix:
            func = func.name.prefix(self._prefix)

        if self._suffix:
            func = func.name.suffix(self._suffix)

        return df.with_columns(func)
