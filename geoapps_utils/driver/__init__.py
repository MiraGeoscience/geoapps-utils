# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2023-2025 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import inspect
import sys
from json import load
from pathlib import Path

from geoapps_utils.driver.driver import BaseDriver


def fetch_driver_class(filepath: str | Path):
    with open(filepath, encoding="utf-8") as jsonfile:
        uijson = load(jsonfile)

    module = __import__(uijson["run_command"], fromlist=[""])

    cls = None
    for _, cls in inspect.getmembers(module):
        try:
            if issubclass(cls, BaseDriver) and cls.__module__ == module.__name__:
                break
        except TypeError:
            continue

    else:
        raise ValueError(f"No valid driver class found in {uijson['run_command']}")

    return cls


if __name__ == "__main__":
    file = sys.argv[1]
    driver_cls = fetch_driver_class(file)
    driver_cls.start(file)
