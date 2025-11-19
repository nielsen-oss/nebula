"""Nebula storage module."""

from typing import Any, Hashable

from nlsn.nebula.logger import logger

__all__ = ["assert_is_hashable", "nebula_storage"]

_PRE: str = "Nebula Storage: "


def assert_is_hashable(o):
    """Assert if an object is hashable."""
    if not isinstance(o, Hashable):
        raise AssertionError(f'{_PRE}"{o}" is not hashable')


class _NebulaStorage:
    def __init__(self):
        """Initialize an empty storage.

        By default, it permits overwriting, and users can enable or disable it
        using the 'allow_overwriting()' and 'disallow_overwriting()' methods.
        Verify the current overwriting permission with the
        'is_overwriting_allowed' property.

        It is also possible to enable or disable the storage debug mode.
        If an object is set with 'debug=True' but the debug mode is disabled,
        the 'set' method will act as a bypass without actually storing the data.
        E.g.
        >>> nebula_storage.set("a", 1)  # will be stored
        >>> nebula_storage.allow_debug(True)  # activate debug
        >>> nebula_storage.set("b", 2, debug=True)  # will be stored
        >>> nebula_storage.allow_debug(False)  # deactivate debug
        >>> nebula_storage.set("c", 3, debug=True)  # will not be stored

        >>> nebula_storage.get("a")
        1
        >>> nebula_storage.get("b")
        2
        >>> nebula_storage.get("c")
        KeyError

        Check the current storage debug mode with the 'is_debug_mode' property.
        Defaults to False.

        Properties:
            is_overwriting_allowed (bool): Whether overwriting is allowed or not.
            debug_mode (bool): Whether the debug is activeor or not.

        Methods:
            allow_overwriting(): Allow overwriting in the storage.
            disallow_overwriting(): Disallow overwriting in the storage.
            allow_debug(v: bool): Activate / deactivate the debug storage mode.
            list_keys(): Return the sorted list of keys in storage.
            count_objects(): Return the number of keys in the cache.
            clear(keys=None): Clear all cache or remove specific keys.
            set(key: str, value, *, debug: bool = False): Add an object to the storage.
            get(key): Get an object from the storage.
            isin(key: str): Check if an object is stored.
        """
        self._n: int = 0
        self._cache: dict[str, Any] = {}
        self._allow_overwrite: bool = True
        self._debug_active: bool = False

    @property
    def is_overwriting_allowed(self) -> bool:
        """Return whether overwriting is allowed or not."""
        return self._allow_overwrite

    def allow_overwriting(self) -> None:
        """Allow overwriting."""
        logger.info(f"{_PRE}allow overwriting.")
        self._allow_overwrite = True

    def disallow_overwriting(self) -> None:
        """Disallow overwriting."""
        logger.info(f"{_PRE}disallow overwriting.")
        self._allow_overwrite = False

    def allow_debug(self, v: bool) -> None:
        """Activate / deactivate debugging storage."""
        if not isinstance(v, bool):
            raise TypeError("'allow_debug' accepts only True / False.")
        if v:
            logger.info(f"{_PRE}activate debug storage.")
        else:
            logger.info(f"{_PRE}deactivate debug storage.")
        self._debug_active = v

    @property
    def is_debug_mode(self) -> bool:
        """Return if the debug mode is active."""
        return self._debug_active

    def list_keys(self) -> list:
        """Return the sorted list of keys in storage."""
        return sorted(self._cache.keys())

    def count_objects(self) -> int:
        """Return the number of keys in cache."""
        return self._n

    def clear(self, keys=None) -> None:
        """Clear all cache or remove some specific keys."""
        if not keys:
            logger.info(f"{_PRE}clear.")
            self._cache.clear()
        elif isinstance(keys, str):
            logger.info(f'{_PRE}clear key "{keys}".')
            self._cache.pop(keys, None)
        elif isinstance(keys, (list, tuple, set, dict)):
            logger.info(f"{_PRE}clear user-defined keys.")
            for key in keys:
                self._cache.pop(key, None)
        self._n = len(self._cache)
        logger.info(f"{_PRE}{self._n} keys remained after clearing.")

    def set(self, key: str, value, *, debug: bool = False) -> None:
        """Add an object to the storage."""
        if (not debug) or (debug and self._debug_active):
            t = type(value)
            logger.info(f'{_PRE}setting an object ({t}) with the key "{key}".')
            assert_is_hashable(key)
            if (not self._allow_overwrite) and (key in self._cache):
                msg = f'{_PRE}key "{key}" already exists and overwriting is disabled.'
                raise KeyError(msg)
            self._cache[key] = value
            self._n = len(self._cache)
        else:
            msg = f'{_PRE}asked to set "{key}" in debug mode but the storage debug '
            msg += "is not active. The object will not be stored."
            logger.info(msg)

    def get(self, key):
        """Get an object in the storage."""
        return self._cache[key]

    def isin(self, key: str) -> bool:
        """Check if an object is stored."""
        return key in self._cache


nebula_storage = _NebulaStorage()
