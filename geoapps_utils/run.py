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
import json
import logging
import sys
from importlib import import_module
from json import load
from pathlib import Path

from geoh5py import Workspace
from geoh5py.groups import UIJsonGroup
from geoh5py.shared.utils import entity2uuid, stringify

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

    if not isinstance(uijson["run_command"], str):
        raise ValueError(
            "'run_command' in ui.json must be a string representing the module path."
            f" Got {type(uijson['run_command'])}."
        )

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


def load_uijson_as_group(
    uijson_path: Path | str,
) -> UIJsonGroup:
    """
    Load a ui.json file as a UIJsonGroup.

    :param uijson_path: Path to a ui.json file.
    :return: UIJsonGroup with options set from the ui.json file.
    """
    uijson_path = Path(uijson_path)
    uijson = load_ui_json(uijson_path)

    if "geoh5" not in uijson:
        raise KeyError(f"ui.json file {uijson_path} must contain a 'geoh5' key.")

    with Workspace(uijson["geoh5"]) as workspace:
        uijson_group = UIJsonGroup.create(
            workspace, name=uijson.get("title", None), options=uijson
        )

    return uijson_group


def run_uijson_group(
    out_group: UIJsonGroup,
    path: Path | str,
    validate: bool = True,
    name: str | None = None,
):
    """
    Run a UIJsonGroup based on its options.

    :param out_group: A UIJsonGroup with options set.
    :param path: Path to save the output .geoh5 and .ui.json files.
    :param validate: Whether to validate the input parameters before running.
        It depends on the driver implementation.
    :param name: Name of the output files. If
    """
    # ensure the group has a driver
    if not isinstance(out_group, UIJsonGroup):
        raise TypeError(
            f"Input 'out_group' must be a UIJsonGroup. Got {type(out_group)}."
        )

    path = Path(path).resolve()

    if not out_group.options:
        raise ValueError("UIJsonGroup must have options set.")

    name = name or out_group.name.replace(" ", "_")
    json_path = path / f"{name}.ui.json"
    h5_path = path / f"{name}.geoh5"

    if h5_path.exists():
        raise FileExistsError(
            f"File {h5_path} already exists."
            "Please remove it or choose another output path."
        )

    json_str = json.dumps(stringify(out_group.options, [entity2uuid]), indent=4)
    json_path.write_text(json_str)

    with Workspace(out_group.options["geoh5"]) as workspace:
        with Workspace.create(h5_path) as new_workspace:
            out_group = workspace.get_entity(out_group.uid)[0]
            out_group.copy(
                parent=new_workspace, copy_children=True, copy_relatives=True
            )
            current_driver = fetch_driver_class(json_path)
            current_driver.start(json_path, validate=validate)


def run_uijson_file(
    uijson_path: Path | str,
    output_path: Path | str,
    validate: bool = True,
    name: str | None = None,
):
    """
    Run a ui.json file.

    :param uijson_path: Path to a ui.json file.
    :param output_path: Path to save the output .geoh5 and .ui.json files.
    :param validate: Whether to validate the input parameters before running.
        It depends on the driver implementation.
    :param name: Name of the output files.
    """
    uijson_group = load_uijson_as_group(uijson_path)
    run_uijson_group(uijson_group, output_path, validate=validate, name=name)


if __name__ == "__main__":
    file = sys.argv[1]
    driver_cls = fetch_driver_class(file)
    driver_cls.start(file)
