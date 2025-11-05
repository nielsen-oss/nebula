"""Example of how transformers could be simplified with Narwhals."""

import narwhals as nw
from typing import Iterable, Optional, Union

class DropColumns:
    """Simplified DropColumns using Narwhals."""
    
    def __init__(
        self,
        *,
        columns: Optional[Union[str, list[str]]] = None,
        # ... other params same as before
    ):
        self.columns = columns
    
    def transform(self, df):
        """Single transform method for all backends."""
        # Convert to Narwhals DataFrame
        nw_df = nw.from_native(df)
        
        # Get columns to drop (using existing selection logic)
        cols_to_drop = self._get_selected_columns(df)
        
        # Drop columns using Narwhals API
        result = nw_df.drop(cols_to_drop)
        
        # Convert back to native format
        return nw.to_native(result)


class RenameColumns:
    """Simplified RenameColumns using Narwhals."""
    
    def transform(self, df):
        nw_df = nw.from_native(df)
        
        # Single rename implementation
        result = nw_df.rename(self._map_rename)
        
        return nw.to_native(result)


class SelectColumns:
    """Simplified SelectColumns using Narwhals."""
    
    def transform(self, df):
        nw_df = nw.from_native(df)
        
        # Get columns to select
        cols_to_select = self._get_selected_columns(df)
        
        # Select columns
        result = nw_df.select(cols_to_select)
        
        return nw.to_native(result)