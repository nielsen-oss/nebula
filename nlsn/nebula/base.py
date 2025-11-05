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
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import narwhals as nw

from nlsn.nebula.auxiliaries import select_columns
from nlsn.nebula.df_types import get_dataframe_type
from nlsn.nebula.metaclasses import InitParamsStorage
from nlsn.nebula.storage import nebula_storage as ns

__all__ = ["LazyWrapper", "Transformer", "nlazy"]


def nlazy(func: FunctionType) -> FunctionType:
    """A decorator to mark a function as 'lazy'."""
    func._n_lazy = True
    return func


class Transformer(metaclass=InitParamsStorage):
    """Base class for transformers."""

    def __init__(self):
        """Initialize the base transformer."""
        self._transformer_init_params: Dict[str, Any] = {}
        self.__columns_selector: Callable
        self._desc: Optional[str] = None

    def _set_columns_selections(
            self,
            *,
            columns: Optional[Union[Iterable[str], str]] = None,
            regex: Optional[str] = None,
            glob: Optional[str] = None,
            startswith: Optional[Union[str, Iterable[str]]] = None,
            endswith: Optional[Union[str, Iterable[str]]] = None,
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

    def _get_selected_columns(self, df) -> List[str]:
        """Return the dataframe requested columns."""
        return self.__columns_selector(list(df.columns))

    @property
    def transformer_init_parameters(self) -> dict:
        """Return the initialization parameters as dict."""
        return deepcopy(self._transformer_init_params)

    def transform(self, df):
        """Public transform method."""
        if isinstance(df, nw.DataFrame):
            # narwhals df in -> narwhals df out
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
    """Determine whether a function is lazy."""
    if not isinstance(o, FunctionType):
        return False
    return getattr(o, "_n_lazy", False)


def is_ns_lazy_request(o) -> bool:
    """Determine whether an object is NS + its key."""
    if isinstance(o, (list, tuple)):
        if len(o) == 2:
            if o[0] is ns:
                return True
    return False


def extract_lazy_params(kwargs: dict) -> dict:
    params = {}
    for k, v in kwargs.items():
        if is_lazy_function(v):
            params[k] = v()
        elif is_ns_lazy_request(v):
            # here v is a 2-element list/tuple
            params[k] = v[0].get(v[1])
        else:
            params[k] = v
    return params


class LazyWrapper:
    """Lazy wrapper class."""

    def __init__(self, trf, **kwargs):
        """Store the transformer class and its initialization parameters."""
        self.trf = trf
        self.kwargs = kwargs

    def transform(self, df):
        """Create the actual object and call the 'transform' method."""
        params: dict = extract_lazy_params(self.kwargs)
        trf = self.trf(**params)
        ret = trf.transform(df)
        return ret
