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

import pytest

from geoapps_utils.utils.logger import get_logger


def test_logger_warning(caplog):
    """
    Test that the logger is set up correctly.
    """
    # test with everything
    logger = get_logger(
        "my-app",
        timestamp=True,
        level_name=True,
        propagate=True,  # will be set to false because level
        add_name=True,
        level="warning",
    )

    with caplog.at_level(logging.WARNING):
        logger.warning("Test log message")

    assert "Test log message" in caplog.text
    assert "my-app" in caplog.text
    assert "WARNING" in caplog.text


def test_logger_info(caplog):
    # test with nothing (expect propagate)
    logger_2 = get_logger(
        timestamp=False,
        level_name=False,
        propagate=True,
        add_name=False,
    )

    with caplog.at_level(logging.INFO):
        logger_2.info("Test log message")

    assert "Test log message" in caplog.text
    assert caplog.records[0].levelname == "INFO"
    assert caplog.records[0].name == "root"


def test_logger_no_propagate(caplog):
    # test with  propagate false
    logger_3 = get_logger(
        "my-app", timestamp=False, level_name=False, propagate=False, add_name=False
    )

    with caplog.at_level(logging.INFO):
        logger_3.info("Test log message")

    assert caplog.text == ""


def test_logger_level_errors():
    with pytest.raises(TypeError, match="Level must be a string or LoggerLevel"):
        get_logger(level=5)  # type: ignore
