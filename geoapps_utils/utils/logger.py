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


def get_logger(
    name: str | None = None,
    *,
    timestamp: bool = False,
    level_name: bool = True,
    propagate: bool = True,
    add_name: bool = True,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a logger with a timestamped stream and specified log level.

    :param name: Name of the logger.
    :param timestamp: Whether to include a timestamp in the log format.
    :param level_name: Whether to include the log level name in the log format.
    :param propagate: Whether to propagate log messages to the parent logger.
    :param add_name: Whether to include the logger name in the log format.
    :param level: Logging level (e.g., logging.DEBUG, logging.INFO).

    :return: Configured logger instance.
    """
    log = logging.getLogger(name)

    log.propagate = propagate
    stream_handler: logging.Handler = logging.StreamHandler()
    if log.handlers:
        if not propagate:
            log.handlers.clear()
        else:
            stream_handler = log.handlers[0]

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

    log.setLevel(level)

    return log
