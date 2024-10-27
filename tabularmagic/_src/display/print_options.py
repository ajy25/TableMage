import logging
from pathlib import Path
import sys


def make_default_logger() -> logging.Logger:
    logger = logging.Logger(name="Default TabularMagic Logger")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger


def make_secondary_logger() -> logging.Logger:
    """Creates a secondary logger for file-logging purposes.
    Unlike the default logger, this logger does not wrap text and does not
    include color/bold formatting.
    """
    logger = logging.Logger(name="No format TabularMagic Logger")
    logger.setLevel(logging.INFO)
    return logger


class _PrintOptions:
    """Class for setting and tracking options for printing and logging."""

    def __init__(self):
        """Initializes the a _PrintOptions object with default settings."""
        self._logger = make_default_logger()
        self._secondary_logger = make_secondary_logger()
        self._muted = False

        self._n_decimals = 3
        self._max_line_width = 88  # consistent with Python Black

    def _log_info(self, msg: str, secondary: bool = False):
        if not self._muted:
            if secondary:
                self._secondary_logger.info(msg)
            else:
                self._logger.info(msg)

    def _log_debug(self, msg: str, secondary: bool = False):
        if not self._muted:
            if secondary:
                self._secondary_logger.debug(msg)
            else:
                self._logger.debug(msg)

    def reset_logger(self, logger: logging.Logger | None = None):
        """Sets a new logger.

        Parameters
        ----------
        logger : logging.Logger | None
            Default : None. If None, resets the logger to the default.
        """
        if logger is None:
            self._logger = make_default_logger()
        else:
            self._logger = logger

    def reset_secondary_logger(self, logger: logging.Logger | None = None):
        """Sets a new secondary logger.

        Parameters
        ----------
        logger : logging.Logger | None
            Default : None. If None, resets the logger to the default.
        """
        if logger is None:
            self._secondary_logger = make_secondary_logger()
        else:
            self._secondary_logger = logger

    def add_log_file(self, path: Path | str, level: int = logging.DEBUG):
        """Adds a file handler to the secondary logger.

        Parameters
        ----------
        path : Path | str
            Path to the .log or .txt file.

        level : int
            Default: logging.DEBUG.
        """
        file_handler = logging.FileHandler(str(path))
        file_handler.setLevel(level)
        self._secondary_logger.addHandler(file_handler)

    def mute(self):
        """Mutes. No messages will be printed or logged."""
        self._muted = True

    def unmute(self):
        """Unmutes. Messages will be printed or logged."""
        self._muted = False


print_options = _PrintOptions()
