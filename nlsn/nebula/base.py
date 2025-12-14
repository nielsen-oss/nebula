"""Define the base transformer class.

This class will be inherited from all nebula transformers.

If user-defined custom transformers are meant to be invoked
from the Nebula pipeline runner, they should import this base class.

The Narwhals implementation has this logic:
Input: Narwhals DF
├─ Has _transform_nw? → Use it directly (nw → nw)
└─ No _transform_nw? → Convert to native → _select_transform() → Convert back to nw
Input: Native DF (pandas/polars/spark)
├─ Has _transform_nw? → Wrap in nw → _transform_nw() → Unwrap to native
└─ No _transform_nw? → Use backend-specific method via _select_transform()
"""

from copy import deepcopy
from functools import partial
from types import FunctionType
from typing import Any, Callable, Iterable

import narwhals as nw

from nlsn.nebula.auxiliaries import select_columns
from nlsn.nebula.df_types import get_dataframe_type
from nlsn.nebula.metaclasses import InitParamsStorage
from nlsn.nebula.storage import nebula_storage as ns

__all__ = ["LazyWrapper", "Transformer", "nlazy"]


def nlazy(func: FunctionType) -> FunctionType:
    """A decorator to mark a function as 'lazy'.

    Use this decorator to mark functions that should be evaluated
    at transform time rather than at pipeline definition time.

    Example:
        >>> @nlazy
        ... def get_current_date():
        ...     return datetime.now().strftime("%Y-%m-%d")
        >>>
        >>> # The function won't be called until transform() is invoked
        >>> lazy_trf = LazyWrapper(AddLiterals, data=[{"alias": "date", "value": get_current_date}])
    """
    func._n_lazy = True
    return func


class Transformer(metaclass=InitParamsStorage):
    """Base class for transformers."""

    def __init__(self):
        """Initialize the base transformer."""
        self._transformer_init_params: dict[str, Any] = {}
        self.__columns_selector: Callable
        self._desc: str | None = None

    def _set_columns_selections(
            self,
            *,
            columns: str | Iterable[str] | None = None,
            regex: str | None = None,
            glob: str | None = None,
            startswith: str | Iterable[str] | None = None,
            endswith: str | Iterable[str] | None = None,
            allow_excess_columns: bool = False,
    ) -> None:
        """Prepare the input for the function 'auxiliaries.select_columns'."""
        self.__columns_selector = partial(
            select_columns,
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
            allow_excess_columns=allow_excess_columns,
        )

    def set_description(self, desc: str) -> None:
        """Set the transformer description."""
        self._desc = desc

    def get_description(self) -> str:
        """Get the transformer description."""
        return self._desc

    def _get_selected_columns(self, df) -> list[str]:
        """Return the dataframe requested columns."""
        return self.__columns_selector(list(df.columns))

    @property
    def transformer_init_parameters(self) -> dict:
        """Return the initialization parameters as dict."""
        return deepcopy(self._transformer_init_params)

    def transform(self, df):
        """Public transform method."""
        if isinstance(df, (nw.DataFrame, nw.LazyFrame)):
            # narwhals in -> narwhals out
            if hasattr(self, '_transform_nw'):  # it's nw compatible
                return self._transform_nw(df)
            else:  # it requires separated logic
                df_native = nw.to_native(df)
                df_out = self._select_transform(df_native)
                return nw.from_native(df_out)

        # native df in -> native df out
        if hasattr(self, '_transform_nw'):  # it's nw compatible
            df_nw = nw.from_native(df)
            df_out = self._transform_nw(df_nw)
            return nw.to_native(df_out)
        # it requires separated logic
        return self._select_transform(df)

    def transform_pandas(self, df):
        """Public transform method for Pandas transformers."""
        return self._transform_pandas(df)

    def transform_polars(self, df):
        """Public transform method for Polars transformers."""
        return self._transform_polars(df)

    def transform_spark(self, df):
        """Public transform method for Spark transformers."""
        return self._transform_spark(df)

    def _select_transform(self, df):
        # Fallback for non-Narwhals transformation
        name: str = get_dataframe_type(df)
        if name == "pandas":
            return self._transform_pandas(df)
        elif name == "polars":
            return self._transform_polars(df)
        elif name == "spark":
            return self._transform_spark(df)
        else:  # pragma: no cover
            raise ValueError(f"Unknown dataframe type {name}")


def is_lazy_function(o) -> bool:
    """Determine whether a function is lazy (decorated with @nlazy)."""
    if not isinstance(o, FunctionType):
        return False
    return getattr(o, "_n_lazy", False)


def is_ns_lazy_request(o) -> bool:
    """Determine whether an object is a nebula storage lazy reference.

    A lazy NS request is a 2-element tuple/list where the first element
    is the nebula_storage instance and the second is the key string.

    Example:
        >>> is_ns_lazy_request((ns, "my_key"))
        True
        >>> is_ns_lazy_request([ns, "my_key"])
        True
        >>> is_ns_lazy_request(("not_ns", "my_key"))
        False
    """
    if isinstance(o, (list, tuple)):
        if len(o) == 2:
            if o[0] is ns:
                return True
    return False


def _resolve_lazy_value(obj):
    """Recursively resolve lazy values in nested structures.

    This function traverses nested dicts, lists, and tuples, resolving:
    - Functions decorated with @nlazy: calls them and uses return value
    - Nebula storage references (ns, "key"): fetches value from storage

    Args:
        obj: Any value that may contain lazy references at any nesting level.

    Returns:
        The resolved value with all lazy references evaluated.

    Example:
        >>> ns.set("config_value", 42)
        >>> @nlazy
        ... def get_timestamp():
        ...     return "2024-01-01"
        >>>
        >>> nested = {
        ...     "static": "hello",
        ...     "from_storage": (ns, "config_value"),
        ...     "from_func": get_timestamp,
        ...     "nested_list": [{"deep": (ns, "config_value")}]
        ... }
        >>> resolved = _resolve_lazy_value(nested)
        >>> resolved
        {
            "static": "hello",
            "from_storage": 42,
            "from_func": "2024-01-01",
            "nested_list": [{"deep": 42}]
        }
    """
    # Priority 1: Check for @nlazy decorated function
    if is_lazy_function(obj):
        return obj()

    # Priority 2: Check for nebula storage reference (ns, "key")
    # IMPORTANT: This must come BEFORE generic tuple handling
    if is_ns_lazy_request(obj):
        return obj[0].get(obj[1])

    # Priority 3: Recurse into dictionaries
    if isinstance(obj, dict):
        return {k: _resolve_lazy_value(v) for k, v in obj.items()}

    # Priority 4: Recurse into lists
    if isinstance(obj, list):
        return [_resolve_lazy_value(item) for item in obj]

    # Priority 5: Recurse into tuples (but NOT ns references - handled above)
    if isinstance(obj, tuple):
        return tuple(_resolve_lazy_value(item) for item in obj)

    # Base case: return value as-is
    return obj


def extract_lazy_params(kwargs: dict) -> dict:
    """Extract and resolve lazy parameters from a kwargs dictionary.

    This function recursively processes all values in the kwargs dict,
    resolving any lazy references (functions or storage keys) found at
    any nesting level.

    Args:
        kwargs: Dictionary of keyword arguments that may contain lazy
                references at any nesting depth.

    Returns:
        New dictionary with all lazy references resolved.

    Example:
        >>> ns.set("threshold", 0.5)
        >>> @nlazy
        ... def get_columns():
        ...     return ["a", "b", "c"]
        >>>
        >>> params = {
        ...     "columns": get_columns,
        ...     "config": {
        ...         "threshold": (ns, "threshold"),
        ...         "static": True
        ...     }
        ... }
        >>> extract_lazy_params(params)
        {
            "columns": ["a", "b", "c"],
            "config": {
                "threshold": 0.5,
                "static": True
            }
        }
    """
    return _resolve_lazy_value(kwargs)


class LazyWrapper:
    """Wrapper for lazy transformer instantiation.

    LazyWrapper defers transformer instantiation until transform() is called.
    This allows parameters to be resolved at runtime rather than at pipeline
    definition time, enabling dynamic configuration based on:
    - Values computed by functions (decorated with @nlazy)
    - Values stored in nebula_storage during earlier pipeline stages

    Lazy references can be nested at any depth within the parameter structure.

    Example:
        >>> # Store a value during pipeline execution
        >>> ns.set("computed_columns", ["col_a", "col_b"])
        >>>
        >>> # Define a lazy function
        >>> @nlazy
        ... def get_threshold():
        ...     return 0.95
        >>>
        >>> # Create lazy transformer with nested references
        >>> lazy_trf = LazyWrapper(
        ...     MyTransformer,
        ...     columns=(ns, "computed_columns"),  # flat reference
        ...     config={
        ...         "threshold": get_threshold,     # nested in dict
        ...         "filters": [
        ...             {"value": (ns, "filter_val")}  # deeply nested
        ...         ]
        ...     }
        ... )
        >>>
        >>> # At transform time, all references are resolved
        >>> result = lazy_trf.transform(df)
    """

    def __init__(self, trf, **kwargs):
        """Store the transformer class and its initialization parameters.

        Args:
            trf: The transformer class (not instance) to instantiate lazily.
            **kwargs: Keyword arguments for the transformer. May contain
                      lazy references at any nesting level.
        """
        self.trf = trf
        self.kwargs = kwargs

    def transform(self, df):
        """Instantiate the transformer with resolved params and transform.

        This method:
        1. Recursively resolves all lazy references in kwargs
        2. Instantiates the transformer with resolved parameters
        3. Calls transform() on the new instance

        Args:
            df: The dataframe to transform.

        Returns:
            The transformed dataframe.
        """
        params: dict = extract_lazy_params(self.kwargs)
        trf = self.trf(**params)
        return trf.transform(df)