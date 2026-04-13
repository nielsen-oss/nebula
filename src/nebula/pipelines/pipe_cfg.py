"""Pipeline configuration."""

__all__ = ["PipelineConfig"]

_DEFAULTS = {
    "max_param_length": 80,
    "verbose": True,
    "show_params": False,
}


class _PipelineConfig:
    """Nebula pipeline configuration.

    Provides validated access to pipeline settings. Keys are fixed to
    prevent silent typos.

    Usage:
        from nebula import PipelineConfig

        PipelineConfig["verbose"] = False       # bracket access
        PipelineConfig.verbose = False          # attribute access
        print(PipelineConfig)                   # show all settings
    """

    def __init__(self, defaults: dict):
        object.__setattr__(self, "_data", dict(defaults))

    def __getitem__(self, key: str):
        self._check_key(key)
        return self._data[key]

    def __setitem__(self, key: str, value):
        self._check_key(key)
        self._data[key] = value

    def __getattr__(self, key: str):
        if key.startswith("_"):
            raise AttributeError(key)
        self._check_key(key)
        return self._data[key]

    def __setattr__(self, key: str, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._check_key(key)
            self._data[key] = value

    def _check_key(self, key: str) -> None:
        if key not in self._data:
            valid = ", ".join(sorted(self._data))
            raise KeyError(f"Unknown config key '{key}'. Valid keys: {valid}")

    def reset(self) -> None:
        """Reset all settings to defaults."""
        self._data.update(_DEFAULTS)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self._data.items()))
        return f"PipelineConfig({items})"


PipelineConfig = _PipelineConfig(_DEFAULTS)
