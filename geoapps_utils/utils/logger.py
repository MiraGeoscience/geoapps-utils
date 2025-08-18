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


def get_logger(name: str | None = None, timestamp: bool = False) -> logging.Logger:
    """
    Get a logger with a timestamped stream and specified log level.

    :param name: Name of the logger.
    :param timestamp: Whether to include a timestamp in the log format.
    """
    log = logging.getLogger(name)

    if log.handlers:
        stream_handler = log.handlers[0]
    else:
        stream_handler = logging.StreamHandler()

    # Set the format for the logger
    formatting = "%(levelname)s"

    if name:
        formatting += " [%(name)s]"

    if timestamp:
        formatting += " %(asctime)s"

    formatter = logging.Formatter(formatting + " %(message)s")
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    return log
