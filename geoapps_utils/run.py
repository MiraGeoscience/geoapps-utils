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
import shutil
import sys
import tempfile
from importlib import import_module
from json import load
from pathlib import Path

from geoh5py import Workspace
from geoh5py.groups import UIJsonGroup
from geoh5py.ui_json import InputFile

from geoapps_utils.base import Driver


logger = logging.getLogger()


def load_ui_json_as_dict(filepath: str | Path | dict) -> dict:
    """
    Load a ui.json file as a dictionary

    :param filepath: Path to a ui.json file.

    :return: Parsed JSON dictionary.
    """

    if isinstance(filepath, (str, Path)):
        with open(filepath, encoding="utf-8") as jsonfile:
            uijson = load(jsonfile)
    else:
        uijson = filepath

    if not isinstance(uijson, dict):
        raise ValueError(f"Invalid ui.json file: {filepath}.")

    return uijson


def fetch_driver_class(json_dict: str | Path | dict) -> type[Driver]:
    """
    Fetch the driver class from the ui.json 'run_command'.

    :param json_dict: Path to a ui.json file with a 'run_command' key.

    :return: Driver class.
    """
    # TODO Remove after deprecation of geoapps_utils.driver
    from geoapps_utils.driver.driver import (  # pylint: disable=import-outside-toplevel, cyclic-import
        BaseDriver,
    )

    uijson = load_ui_json_as_dict(json_dict)

    if "run_command" not in uijson or not isinstance(uijson["run_command"], str):
        raise KeyError(
            "'run_command' in ui.json must be a string representing the module path."
            f" Got {uijson.get('run_command', None)}."
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


def run_uijson_group(out_group: UIJsonGroup) -> Driver:
    """
    Run the function defined in the 'out_group' UIJsonGroup.

    This function must be call in the uijson_group context.

    :param out_group: A UIJsonGroup with options set.

    :return: Driver instance.
    """
    if not out_group.options:
        raise ValueError("UIJsonGroup must have options set.")

    # create a uijson file from options
    uijson_file = out_group.add_ui_json()

    # create a temporary uijson file
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        tmpdir_uijson = uijson_file.save_file(tmpdir, name=f"{out_group.name}.ui.json")

        # run the temporary file
        driver_class = fetch_driver_class(tmpdir_uijson)
        driver_instance = driver_class.start(tmpdir_uijson)

    return driver_instance


def get_new_workspace_path(
    name: str, destination: str | Path, new_workspace_name: str | None = None
) -> Path:
    """
    Check that a workspace does not already exist at the destination.

    :param name: Original geoh5 file name.
    :param destination: Path to copy the ui.json file to.
    :param new_workspace_name: New geoh5 file name. If None, the original name is kept.

    :raises FileExistsError: If the workspace already exists.

    :return: The path to the new workspace.
    """
    new_workspace_name = new_workspace_name or name
    workspace_path = Path(destination) / new_workspace_name
    workspace_path = workspace_path.with_suffix(".geoh5")

    if workspace_path.is_file():
        raise FileExistsError(f"File {workspace_path} already exists.")

    return workspace_path


def copy_uijson_relatives_only(
    uijson_path: Path | str,
    destination: Path | str,
    new_workspace_name: str | None = None,
    monitoring_directory: Path | str | None = None,
) -> Path:
    """
    Copy a ui.json file to a new location, optionally changing the geoh5 file name.

    :param uijson_path: Path to a ui.json file.
    :param destination: Path to copy the ui.json file to.
    :param new_workspace_name: New geoh5 file name. If None, the original name is kept.
    :param monitoring_directory: New monitoring directory. If None, the original is kept.
    """
    ifile = InputFile.read_ui_json(uijson_path)

    workspace_path = get_new_workspace_path(ifile.name, destination, new_workspace_name)

    with ifile.geoh5.open():
        with Workspace.create(workspace_path) as new_workspace:
            ifile.copy_relatives(new_workspace)
            temp_json = ifile.ui_json.copy()
            temp_json["geoh5"] = workspace_path
            if monitoring_directory is not None:
                temp_json["monitoring_directory"] = str(monitoring_directory)
            new_input_file = InputFile(ui_json=temp_json)

            uijson_path_name = new_input_file.write_ui_json(path=str(destination))

    return uijson_path_name


def copy_uijson_and_workspace(
    uijson_path: Path | str,
    destination: Path | str,
    new_workspace_name: str | None = None,
    monitoring_directory: Path | str | None = None,
):
    """
    Copy a ui.json file and its geoh5 file to a new location.

    Optionally changing the geoh5 file name.
    Warning: This function copies the entire geoh5 file, which may be large.

    :param uijson_path: Path to a ui.json file.
    :param destination: Path to copy the ui.json file to.
    :param new_workspace_name: New geoh5 file name. If None, the original name is kept.
    :param monitoring_directory: New monitoring directory. If None, the original is kept.

    :return: Path to the new ui.json file.
    """
    ifile = InputFile.read_ui_json(uijson_path)
    workspace_path = get_new_workspace_path(ifile.name, destination, new_workspace_name)
    shutil.copy(ifile.geoh5.h5file, workspace_path)

    with Workspace(workspace_path):
        temp_json = ifile.ui_json.copy()
        temp_json["geoh5"] = workspace_path
        if monitoring_directory is not None:
            temp_json["monitoring_directory"] = str(monitoring_directory)
        new_input_file = InputFile(ui_json=temp_json)

        uijson_path_name = new_input_file.write_ui_json(path=str(destination))

    return uijson_path_name


def run_from_outgroup_name(
    workspace_path: str | Path,
    name: str,
    *,
    destination: Path | str | None = None,
    new_workspace_name: str | None = None,
    monitoring_directory: Path | str | None = None,
) -> Driver:
    """
    Run the function defined in the 'out_group' UIJsonGroup.

    :param workspace_path: Path to a geoh5 file.
    :param name: Name or str UUID of a UIJsonGroup with options set.
    :param destination: Path to copy the ui.json file to. If None, a temporary directory is used.
    :param new_workspace_name: New geoh5 file name. If None, the original name is kept.
    :param monitoring_directory: New monitoring directory. If None, the original is kept.

    :return: Driver instance.
    """
    if not Path(workspace_path).is_file():
        raise FileNotFoundError(f"Workspace file '{workspace_path}' does not exist.")

    with Workspace(workspace_path) as workspace:
        out_group = workspace.get_entity(name)[0]

        if not isinstance(out_group, UIJsonGroup):
            raise TypeError(
                f"Entity with name or id '{name}' "
                f"is not a UIJsonGroup. Got {type(out_group)}."
            )

        if destination is not None:
            workspace_path = get_new_workspace_path(
                out_group.name, destination, new_workspace_name
            )
            with Workspace.create(workspace_path) as new_workspace:
                copy_out_group = out_group.copy(
                    new_workspace, copy_children=False, copy_relatives=True
                )

                if copy_out_group is None:  # pragma: no cover
                    raise RuntimeError("Failed to copy the UIJsonGroup.")

                if monitoring_directory is not None:
                    copy_out_group.modify_option(
                        "monitoring_directory", str(monitoring_directory)
                    )

                return run_uijson_group(copy_out_group)

        return run_uijson_group(out_group)


def run_from_uijson(
    uijson_path: str | Path,
    *,
    destination: Path | str | None = None,
    new_workspace_name: str | None = None,
    monitoring_directory: Path | str | None = None,
    relatives_only: bool = True,
) -> Driver:
    """
    Run a ui.json file, optionally copying it to a new location and changing the geoh5 file name.

    :param uijson_path: Path to a ui.json file.
    :param destination: Path to copy the ui.json file to. If None, a temporary directory is used.
    :param new_workspace_name: New geoh5 file name. If None, the original name is kept.
    :param monitoring_directory: New monitoring directory. If None, the original is kept.
    :param relatives_only: If True, only copy the entities referenced in the ui.json file.
                           If False, copy the entire geoh5 file. Default is True.

    :return: Driver instance.
    """
    if destination is not None:
        if relatives_only:
            uijson_path = copy_uijson_relatives_only(
                uijson_path,
                destination,
                new_workspace_name=new_workspace_name,
                monitoring_directory=monitoring_directory,
            )
        else:
            uijson_path = copy_uijson_and_workspace(
                uijson_path,
                destination,
                new_workspace_name=new_workspace_name,
                monitoring_directory=monitoring_directory,
            )

    driver_class = fetch_driver_class(uijson_path)
    driver_instance = driver_class.start(uijson_path)

    return driver_instance


if __name__ == "__main__":
    file = sys.argv[1]
    driver_cls = fetch_driver_class(file)
    driver_cls.start(file)
