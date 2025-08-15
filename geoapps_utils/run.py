# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2023-2025 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import inspect
import logging
import sys
from importlib import import_module
from json import load
from pathlib import Path

from geoapps_utils.base import Driver


logger = logging.getLogger()


def load_ui_json(filepath: str | Path | dict) -> dict:
    """
    Load a ui.json file.

    :param filepath: Path to a ui.json file.
    :return: Parsed JSON dictionary.
    """

    if isinstance(filepath, (str, Path)):
        with open(filepath, encoding="utf-8") as jsonfile:
            uijson = load(jsonfile)
    else:
        uijson = filepath

    if not isinstance(uijson, dict) or "run_command" not in uijson:
        raise ValueError(
            f"Invalid ui.json file: {filepath}. It must contain a 'run_command' key."
        )

    return uijson


def fetch_driver_class(json_dict: str | Path | dict) -> type[Driver]:
    """
    Fetch the driver class from the ui.json 'run_command'.

    :param filepath: Path to a ui.json file with a 'run_command' key.
    """
    # TODO Remove after deprecation of geoapps_utils.driver

    from geoapps_utils.driver.driver import (  # pylint: disable=import-outside-toplevel, cyclic-import
        BaseDriver,
    )

    uijson = load_ui_json(json_dict)
    module = import_module(uijson["run_command"])
    cls = None
    for _, cls in inspect.getmembers(module):
        try:
            if (
                issubclass(cls, Driver | BaseDriver)
                and cls.__module__ == module.__name__
            ):
                break
        except TypeError:
            continue

    else:
        logger.warning(
            "\n\nApplicationError: No valid driver class found in module %s\n\n",
            uijson["run_command"],
        )
        sys.exit(1)

    return cls


if __name__ == "__main__":
    file = sys.argv[1]
    driver_cls = fetch_driver_class(file)
    driver_cls.start(file)
