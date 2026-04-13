"""Logger module."""

import logging

__all__ = ["logger"]

# _fmt = "%(asctime)s | %(filename)s:%(lineno)s [%(levelname)s]: %(message)s "
_fmt = "%(asctime)s | [%(levelname)s]: %(message)s "


class Logger:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format=_fmt)
        self._logger = logging.getLogger(__name__)
        self._verbose: bool = True

    @property
    def verbose(self) -> bool:
        """Return whether verbose logging is enabled."""
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Enable or disable verbose logging."""
        self._verbose = value

    def info(self, msg, *args, **kwargs):
        """Log info message only when verbose is enabled."""
        if self._verbose:
            self._logger.info(msg, *args, **kwargs)

    def __getattr__(self, item):
        """__getattr__ implementation.

        Checks methods and attributes in the self.logger class
        if not found in the LoggingHelper class
        reimplemented the search in self for direct calls of __getattr__
        """
        if hasattr(self._logger, item):
            return getattr(self._logger, item)
        else:
            raise AttributeError(f"No attribute {item}")

    @staticmethod
    def set_level(level):  # pragma: no cover
        """Set a new logger level."""
        if isinstance(level, str):
            level = level.upper()
        logging.getLogger().setLevel(level)

    def inject_logger(self, custom_logger: logging.Logger):  # pragma: no cover
        """Set a custom logger.

        Experimental.
        """
        self._logger = custom_logger


logger = Logger()
