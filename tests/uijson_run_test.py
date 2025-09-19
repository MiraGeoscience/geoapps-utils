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

import json
from uuid import UUID

import pytest
from geoh5py import Workspace
from geoh5py.data import Data
from geoh5py.groups import UIJsonGroup
from geoh5py.objects import Points
from geoh5py.ui_json.input_file import InputFile

from geoapps_utils.run import (
    copy_out_group,
    get_new_workspace_name,
    load_ui_json_as_dict,
    run_from_outgroup_name,
    run_from_uijson,
    run_uijson_group,
)

from .dummy_driver_test import TestOptions


# pylint: disable=duplicate-code
def create_uijson(tmp_path):
    with Workspace.create(tmp_path / "original.geoh5") as workspace:
        points = Points.create(workspace)

        out_group = UIJsonGroup.create(workspace, name="uijson_test")

        # Create params
        test_params = {
            "monitoring_directory": None,
            "workspace_geoh5": None,
            "geoh5": workspace,
            "run_command": "tests.dummy_driver_test",
            "title": "test_title",
            "conda_environment": None,
            "conda_environment_boolean": False,
            "generate_sweep": False,
            "workspace": None,
            "run_command_boolean": False,
            "nested_model": {"client": points},
        }

        ui_json = {
            "version": "0.0.0",
            "title": "test_title",
            "conda_environment": "myenv",
            "run_command": "tests.dummy_driver_test",
            "geoh5": workspace,
            "monitoring_directory": None,
            "workspace_geoh5": None,
            "client": {
                "meshType": [UUID("202c5db1-a56d-4004-9cad-baafd8899406")],
                "main": True,
                "label": "Destination",
                "value": points,
                "group": "Objects",
            },
            "out_group": {
                "group": "Output preferences",
                "label": "UIJson group",
                "value": out_group,
                "groupType": "{BB50AC61-A657-4926-9C82-067658E246A0}",
                "visible": True,
                "optional": True,
                "enabled": True,
            },
        }

        out_group.options = ui_json

        params = TestOptions.build(test_params)
        params._input_file = InputFile(ui_json=ui_json)  # pylint: disable=protected-access
        uijson_path = params.write_ui_json(path=tmp_path)

    return uijson_path


def test_run_from_uijson(tmp_path):
    uijson_path = create_uijson(tmp_path)

    monitoring_directory = tmp_path / "monitoring"
    monitoring_directory.mkdir(exist_ok=True)

    destination = tmp_path / "copy"
    destination.mkdir(exist_ok=True)

    run_from_uijson(
        uijson_path, destination=destination, monitoring_directory=monitoring_directory
    )

    # test destination
    with Workspace(destination / "test_run_from_uijson0.ui.geoh5") as workspace:
        assert isinstance(workspace.get_entity("mean_xyz")[0], Data)

    ui_json_file = destination / "test_title.ui.json"
    with open(ui_json_file, encoding="utf-8") as file:
        ui_json_file = file.read()
        ui_json = json.loads(ui_json_file)

    assert ui_json["geoh5"].endswith("test_run_from_uijson0.ui.geoh5")
    assert ui_json["monitoring_directory"].endswith(str(monitoring_directory))

    # test monitoring directory
    monitor_geoh5 = next(iter(monitoring_directory.glob("*.geoh5")))
    with Workspace(monitoring_directory / monitor_geoh5) as workspace:
        assert isinstance(workspace.get_entity("mean_xyz")[0], Data)


def test_run_from_out_group(tmp_path):
    create_uijson(tmp_path)

    monitoring_directory = tmp_path / "monitoring"
    monitoring_directory.mkdir(exist_ok=True)

    destination = tmp_path / "copy"
    destination.mkdir(exist_ok=True)

    run_from_outgroup_name(
        tmp_path / "original.geoh5",
        "uijson_test",
        destination=destination,
        monitoring_directory=monitoring_directory,
        new_workspace_name="testing.geoh5",
    )

    # test destination
    with Workspace(destination / "testing.geoh5") as workspace:
        assert isinstance(workspace.get_entity("mean_xyz")[0], Data)

    # test monitoring directory
    monitor_geoh5 = next(iter(monitoring_directory.glob("*.geoh5")))
    with Workspace(monitoring_directory / monitor_geoh5) as workspace:
        assert isinstance(workspace.get_entity("mean_xyz")[0], Data)


def test_out_group_errors(tmp_path):
    create_uijson(tmp_path)

    with Workspace(tmp_path / "original.geoh5") as workspace:
        # create an empty uijson group
        empty_group = UIJsonGroup.create(workspace, name="empty")

        with pytest.raises(TypeError, match="Input 'out_group' must be"):
            run_uijson_group("bidon")  # type: ignore

        with pytest.raises(ValueError, match="UIJsonGroup must have options"):
            run_uijson_group(empty_group)

        with pytest.raises(TypeError, match="Input 'out_group' must be"):
            copy_out_group("bidon", tmp_path / "copy")  # type: ignore

        with pytest.raises(FileNotFoundError, match="Workspace file"):
            run_from_outgroup_name(tmp_path / "non_existent.geoh5", "uijson_test")

        with pytest.raises(TypeError, match="Entity with name"):
            run_from_outgroup_name(
                tmp_path / "original.geoh5",
                "inexistent",  # type: ignore
            )


def test_utils_errors(tmp_path):
    create_uijson(tmp_path)

    with pytest.raises(ValueError, match="Invalid ui.json file"):
        load_ui_json_as_dict(123)  # type: ignore

    with pytest.raises(FileExistsError, match="File "):
        get_new_workspace_name("original.geoh5", tmp_path)


# def test_load_fetch_uijson_errors(tmp_path):
#     with pytest.raises(ValueError, match="Invalid ui.json file: "):
#         load_ui_json_as_dict(123)  # type: ignore
#
#     uijson = {
#         "title": "empty",
#     }
#
#     ui_json_path_1 = tmp_path / "temp1.ui.json"
#     with open(ui_json_path_1, "w", encoding="utf-8") as file:
#         json.dump(uijson, file)
#
#     with pytest.raises(KeyError, match="'run_command' in ui.json must be a string"):
#         fetch_driver_class(ui_json_path_1)
#
#     uijson["run_command"] = "geoapps_utils.driver"
#
#     ui_json_path_2 = tmp_path / "temp2.ui.json"
#     with open(ui_json_path_2, "w", encoding="utf-8") as file:
#         json.dump(uijson, file)
#
#     with pytest.raises(TypeError, match="Input 'out_group' must be"):
#         run_uijson_group("bidon", path=tmp_path)  # type: ignore
#
#     h5file = tmp_path / "test.geoh5"
#     workspace = Workspace.create(h5file)
#     ui_json_group = UIJsonGroup.create(workspace, name="test")
#
#     with pytest.raises(ValueError, match="UIJsonGroup must have options"):
#         run_uijson_group(ui_json_group, path=tmp_path)
#
#     ui_json_group.options = uijson
#
#     with pytest.raises(FileExistsError, match="File "):
#         run_uijson_group(ui_json_group, path=tmp_path)
