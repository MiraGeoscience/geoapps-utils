# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                          '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from __future__ import annotations

import logging
from enum import Enum


class LoggerLevel(str, Enum):
    """
    The different possible log levels.
    """

    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        """
        Get the current state of the logger.
        """
        if self == LoggerLevel.WARNING:
            return logging.WARNING
        if self == LoggerLevel.INFO:
            return logging.INFO
        if self == LoggerLevel.DEBUG:
            return logging.DEBUG
        if self == LoggerLevel.ERROR:
            return logging.ERROR
        if self == LoggerLevel.CRITICAL:
            return logging.CRITICAL
        return logging.NOTSET

    @classmethod
    def get_logger(cls, level: str | LoggerLevel) -> int:
        """
        Get the logger level from a string or LoggerLevel.

        :param level: The log level as a string or LoggerLevel.

        :return: The corresponding logging level.
        """
        if isinstance(level, str):
            level = cls(level.lower())
        if not isinstance(level, cls):
            raise TypeError(f"Level must be a string or LoggerLevel, got {type(level)}")
        return level.level


def get_logger(
    name: str | None = None,
    *,
    timestamp: bool = False,
    level_name: bool = True,
    propagate: bool | None = None,
    add_name: bool = True,
    level: str | LoggerLevel | None = None,
) -> logging.Logger:
    """
    Get a logger with a timestamped stream and specified log level.

    :param name: Name of the logger.
    :param timestamp: Whether to include a timestamp in the log format.
    :param level_name: Whether to include the log level name in the log format.
    :param propagate: Whether to propagate log messages to the parent logger.
    :param add_name: Whether to include the logger name in the log format.
    :param level: Logging level to use.

    :return: Configured logger instance.
    """
    log = logging.getLogger(name)

    if log.handlers:
        stream_handler = log.handlers[0]
    else:
        stream_handler = logging.StreamHandler()

    # Set the format for the logger
    formatting = ""
    if level_name:
        formatting = "%(levelname)s "

    if name and add_name:
        formatting += "[%(name)s] "

    if timestamp:
        formatting += "%(asctime)s "

    formatter = logging.Formatter(formatting + "%(message)s")
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    if level:
        log.setLevel(LoggerLevel.get_logger(level))
        log.propagate = False
    elif propagate is not None:
        log.propagate = propagate

    return log
