"""DataFrame and column-level operations."""

from typing import Any, Iterable

import narwhals as nw

from nebula.auxiliaries import assert_allowed
from nebula.base import Transformer

__all__ = [
    "DataFrameMethod",
    "HorizontalFunction",
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

# -- Polars-native registries (optional) ------------------------------------
try:
    import polars as pl

    _PL_DATAFRAME_METHODS: set[str] = _get_public_methods(pl.DataFrame)
    _PL_LAZYFRAME_METHODS: set[str] = _get_public_methods(pl.LazyFrame)
    _PL_DF_METHODS: set[str] = _PL_DATAFRAME_METHODS | _PL_LAZYFRAME_METHODS

    _PL_FLAT_COL_METHODS: set[str] = _get_public_methods(pl.col("_"))
    _PL_COL_ACCESSORS: set[str] = {"str", "dt", "list", "struct", "cat", "arr"}
    _PL_NESTED_COL_METHODS: dict[str, set[str]] = {
        acc: _get_public_methods(getattr(pl.col("_"), acc)) for acc in _PL_COL_ACCESSORS
    }
    _HAS_POLARS = True
except ImportError:
    _PL_DATAFRAME_METHODS = set()
    _PL_LAZYFRAME_METHODS = set()
    _PL_DF_METHODS = set()
    _PL_FLAT_COL_METHODS = set()
    _PL_COL_ACCESSORS = set()
    _PL_NESTED_COL_METHODS = {}
    _HAS_POLARS = False


def _is_polars_native(native) -> bool:
    if not _HAS_POLARS:
        return False
    return isinstance(native, (pl.DataFrame, pl.LazyFrame))


_HORIZONTAL_FUNCTIONS: set[str] = {
    "coalesce",
    "concat_str",
    "max_horizontal",
    "min_horizontal",
    "mean_horizontal",
    "sum_horizontal",
    "all_horizontal",
    "any_horizontal",
    "format",
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
        assert_allowed(method, _NW_DF_METHODS | _PL_DF_METHODS, "dataframe-method")

        super().__init__()
        self._method_name: str = method
        self._args: list = args if args else []
        self._kwargs: dict[str, Any] = kwargs if kwargs else {}

    def _transform_nw(self, df):
        """Call the specified method on the Narwhals DataFrame or LazyFrame.

        Tries the Narwhals API first. If the method only exists in Polars
        and the underlying frame is a Polars frame, falls back to calling
        the method on the native Polars object.
        """
        try:
            if isinstance(df, nw.LazyFrame):
                if self._method_name not in _NW_LAZYFRAME_METHODS:
                    raise AttributeError(self._method_name)
            elif self._method_name not in _NW_DATAFRAME_METHODS:
                raise AttributeError(self._method_name)

            method = getattr(df, self._method_name)
            return method(*self._args, **self._kwargs)
        except AttributeError:
            native = nw.to_native(df)
            if not _is_polars_native(native):
                raise
            result = getattr(native, self._method_name)(*self._args, **self._kwargs)
            return nw.from_native(result)


class HorizontalFunction(Transformer):
    def __init__(  # noqa: PLR0913
        self,
        *,
        output_col: str,
        function: str,
        columns: list[str] | None = None,
        regex: str | None = None,
        glob: str | None = None,
        startswith: str | Iterable[str] | None = None,
        endswith: str | Iterable[str] | None = None,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        """Apply a Narwhals top-level function across columns.

        These functions operate horizontally (across columns) rather than
        vertically (within columns).

        Args:
            output_col: Name of the output column
            function: Narwhals function name (e.g., 'coalesce', 'max_horizontal')
            columns: Column name(s) to include
            regex: Regex pattern for column selection
            glob: Glob pattern for column selection
            startswith: Select columns starting with string(s)
            endswith: Select columns ending with string(s)
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function

        Examples:
            >>> # Coalesce multiple email columns
            >>> HorizontalFunction(
            ...     output_col="best_email",
            ...     function="coalesce",
            ...     columns=["email_primary", "email_secondary", "email_backup"]
            ... )

            >>> # Concatenate name parts with space
            >>> HorizontalFunction(
            ...     output_col="full_name",
            ...     function="concat_str",
            ...     columns=["first_name", "middle_name", "last_name"],
            ...     kwargs={"separator": " ", "ignore_nulls": True}
            ... )

            >>> # Get maximum value across metrics
            >>> HorizontalFunction(
            ...     output_col="max_score",
            ...     function="max_horizontal",
            ...     regex="^score_.*"
            ... )

            >>> # Sum all revenue columns
            >>> HorizontalFunction(
            ...     output_col="total_revenue",
            ...     function="sum_horizontal",
            ...     glob="revenue_*"
            ... )
        """
        super().__init__()
        assert_allowed(function, _HORIZONTAL_FUNCTIONS, "horizontal-function")

        self._output_col: str = output_col
        self._function: str = function
        self._args: list = args if args else []
        self._kwargs: dict[str, Any] = kwargs if kwargs else {}

        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )

    def _transform_nw(self, df):
        selection: list[str] = self._get_selected_columns(df)

        if not selection:
            return df

        func = getattr(nw, self._function)
        col_exprs = [nw.col(c) for c in selection]
        result_expr = func(*col_exprs, *self._args, **self._kwargs)
        return df.with_columns(result_expr.alias(self._output_col))


class WithColumns(Transformer):
    def __init__(  # noqa: PLR0913
        self,
        *,
        columns: str | list[str] | None = None,
        regex: str | None = None,
        glob: str | None = None,
        startswith: str | Iterable[str] | None = None,
        endswith: str | Iterable[str] | None = None,
        method: str,
        alias: str | None = None,
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
            alias (str | None):
                If provided, create a new column to store the output of
                the operation. Defaults to None.
            args:
                Positional arguments for the method.
            kwargs:
                Keyword arguments for the method.
            prefix (str | None):
                Add prefix to output column names.
            suffix (str | None):
                Add suffix to output column names.
        """
        if alias:
            if (prefix is not None) or (suffix is not None):
                raise AssertionError("'prefix'/'suffix' cannot be provided with 'alias'")

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
        _all_accessors = {"str", "dt"} | _PL_COL_ACCESSORS
        if n_split == 1:
            assert_allowed(method, _NW_FLAT_COL_METHODS | _PL_FLAT_COL_METHODS, "column-method")
            self._method_accessor = None
            self._method_name = method_splits[0]
        elif n_split == 2:  # noqa: PLR2004
            accessor: str = method_splits[0]
            assert_allowed(accessor, _all_accessors, "column-accessor")
            allowed = _NW_NESTED_COL_METHODS.get(accessor, set()) | _PL_NESTED_COL_METHODS.get(accessor, set())
            assert_allowed(method_splits[1], allowed, "column-method")
            self._method_accessor = accessor
            self._method_name = method_splits[1]
        else:
            raise ValueError(f"Method '{method}' has too many parts. Expected format: 'method' or 'namespace.method'")

        self._args: list = args if args else []
        self._kwargs: dict[str, Any] = kwargs if kwargs else {}
        self._alias: str | None = alias
        self._prefix: str | None = prefix
        self._suffix: str | None = suffix

    def _transform_nw(self, df):
        selection: list[str] = self._get_selected_columns(df)

        if not selection:
            return df

        try:
            meth = nw.col(*selection)
            if self._method_accessor:
                meth = getattr(meth, self._method_accessor)
            meth = getattr(meth, self._method_name)
            func = meth(*self._args, **self._kwargs)

            if self._alias:
                func = func.alias(self._alias)
            if self._prefix:
                func = func.name.prefix(self._prefix)
            if self._suffix:
                func = func.name.suffix(self._suffix)

            return df.with_columns(func)
        except AttributeError:
            native = nw.to_native(df)
            if not _is_polars_native(native):
                raise

            meth = pl.col(*selection)
            if self._method_accessor:
                meth = getattr(meth, self._method_accessor)
            meth = getattr(meth, self._method_name)
            func = meth(*self._args, **self._kwargs)

            if self._alias:
                func = func.alias(self._alias)
            if self._prefix:
                func = func.name.prefix(self._prefix)
            if self._suffix:
                func = func.name.suffix(self._suffix)

            return nw.from_native(native.with_columns(func))
