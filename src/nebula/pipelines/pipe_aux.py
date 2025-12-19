"""Auxiliaries module for pipelines."""

from enum import Enum
from typing import Union

__all__ = [
    "MSG_ACTIVATE_DEBUG_MODE",
    "MSG_DEACTIVATE_DEBUG_MODE",
    "MSG_NOT_UNDERSTOOD",
    "Node",
    "NodeType",
    "StoreRequest",
    "get_replace_with_stored_df_msg",
    "get_store_debug_key_msg",
    "get_store_key_msg",
    "parse_storage_request",
]

MSG_ACTIVATE_DEBUG_MODE: str = "Activate storage debug mode"
MSG_DEACTIVATE_DEBUG_MODE: str = "Deactivate storage debug mode"

MSG_NOT_UNDERSTOOD: str = """
Not understood. At this stage the pipeline is expecting:
- a Transformer object (a <dict> w/ 'transformer' in keys)
- another Pipeline (a <dict> w/ 'pipeline' in keys)
- a storage request like:
    - {"store": "your df name"}
    - {"store_debug": "your df name"}
    - {"storage_debug_mode": True/False}
    - {"replace_with_stored_df": "your df name"}
""".strip()


class NodeType(Enum):
    TRANSFORMER = 0
    LINEAR_PIPELINE = 1
    SPLIT_PIPELINE = 2
    STORE = 3
    STORE_DEBUG = 4
    STORAGE_DEBUG_MODE = 5
    REPLACE_WITH_STORED_DF = 6
    SPLIT_FUNCTION = 7
    SPLIT_NAME = 8
    SPLIT_MERGE = 9
    BRANCH_PRIMARY_DF = 10
    BRANCH_SECONDARY_DF = 11
    BRANCH_MERGE = 12
    REPARTITION_OUTPUT_TO_ORIGINAL = 13
    COALESCE_OUTPUT_TO_ORIGINAL = 14
    APPLY_TO_ROWS = 15
    MERGE_APPLY_TO_ROWS = 16
    OTHERWISE = 17
    INPUT_DF = 18
    OUTPUT_DF = 19


class Node:
    """Node class for the dag."""

    def __init__(self, data, *, t: NodeType, n: int, kwargs=None):
        """Initialize the node.

        Args:
            t (NodeType):
                Node type.
            n (int):
                Node number. It's its identifier, not something that
                must be increased / decreased.
            data (dict(str, any) | None):
                Useful data for the dag visualization.
        """
        self.data = data
        self.t: NodeType = t
        self.kws = kwargs
        self.children: list[Union["Node", dict]] = []
        self.n: int = n

    def add_child(self, o):
        """Add the child to the dag."""
        self.children.append(o)


class StoreRequest(Enum):
    NULL = 0
    STORE_DF = 1
    STORE_DF_DEBUG = 2
    ACTIVATE_DEBUG = 3
    DEACTIVATE_DEBUG = 4
    REPLACE_WITH_STORED_DF = 5


def get_replace_with_stored_df_msg(d: dict) -> tuple[str, str]:
    """Create the message to visualize for a 'replace_with_stored_df' request."""
    key = d["replace_with_stored_df"]
    msg = f'Replace the main dataframe with the one stored with the key "{key}"'
    return key, msg


def get_store_debug_key_msg(d: dict) -> tuple[str, str]:
    """Create the message to visualize for a store-debug request."""
    key = d["store_debug"]
    msg = f'Store the dataframe with the key "{key}" in debug mode'
    return key, msg


def get_store_key_msg(d: dict) -> tuple[str, str]:
    """Create the message to visualize for a store request."""
    key = d["store"]
    msg = f'Store the dataframe with the key "{key}"'
    return key, msg


def parse_storage_request(o) -> StoreRequest:
    """Checks if the given object represents a storage request and parse it.

    A storage request is a <dictionary<str>,<str>> with only
    one key: "store", i.e. {"store": "df_processed"}.

    Args:
        o (any): The object to be checked.

    Returns (bool):
        - True if the object is related to nebula_storage, False otherwise.
        - True if the object must be stored, False otherwise.
        - True if the object must be stored in debug mode, False otherwise.

    Raises:
         TypeError: If the value associated with the request is wrong.
    """
    if not isinstance(o, dict):
        return StoreRequest.NULL

    # It is a dictionary

    if len(o) != 1:  # Wrong length
        return StoreRequest.NULL

    # It is a dictionary of length == 1
    key, value = list(o.items())[0]
    if key == "store":
        if not isinstance(value, str):
            raise TypeError("The value of 'store' must be <str>.")
        # it is a storage request
        return StoreRequest.STORE_DF
    elif key == "store_debug":
        if not isinstance(value, str):
            raise TypeError("The value of 'store_debug' must be <str>.")
        # it is a storage request in debug mode
        return StoreRequest.STORE_DF_DEBUG
    elif key == "storage_debug_mode":
        if not isinstance(value, bool):
            raise TypeError("The value of 'storage_debug_mode' must be <bool>.")
        if value:
            return StoreRequest.ACTIVATE_DEBUG
        else:
            return StoreRequest.DEACTIVATE_DEBUG
    elif key == "replace_with_stored_df":
        if not isinstance(value, str):
            raise TypeError("The value of 'replace_with_stored_df' must be <str>.")
        return StoreRequest.REPLACE_WITH_STORED_DF

    # I should never be here
    return StoreRequest.NULL  # pragma: no cover
