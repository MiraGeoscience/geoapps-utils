# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2022-2026 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from __future__ import annotations

import json

import pytest
from geoh5py import Workspace
from geoh5py.data import Data
from geoh5py.groups import UIJsonGroup

from geoapps_utils.run import (
    get_new_workspace_path,
    load_ui_json_as_dict,
    run_from_outgroup_name,
    run_from_uijson,
    run_uijson_group,
)


# pylint: disable=unused-argument


def test_run_from_uijson(tmp_path, uijson_path):
    monitoring_directory = tmp_path / "monitoring"
    monitoring_directory.mkdir(exist_ok=True)

    destination = tmp_path / "copy"
    destination.mkdir(exist_ok=True)

    driver = run_from_uijson(
        uijson_path, destination=destination, monitoring_directory=monitoring_directory
    )

    # Check it is updatable
    driver.params.update_out_group_options()

    # test destination
    with Workspace(destination / "original.geoh5") as workspace:
        assert isinstance(workspace.get_entity("mean_xyz")[0], Data)

    ui_json_file = destination / "original.ui.json"
    with open(ui_json_file, encoding="utf-8") as file:
        ui_json_file = file.read()
        ui_json = json.loads(ui_json_file)

    assert ui_json["geoh5"].endswith("original.geoh5")
    assert ui_json["monitoring_directory"].endswith(str(monitoring_directory))

    # test monitoring directory
    monitor_geoh5 = next(iter(monitoring_directory.glob("*.geoh5")))
    with Workspace(monitoring_directory / monitor_geoh5) as workspace:
        assert isinstance(workspace.get_entity("mean_xyz")[0], Data)


def test_run_from_uijson_shutil(tmp_path, uijson_path):

    monitoring_directory = tmp_path / "monitoring"
    monitoring_directory.mkdir(exist_ok=True)

    destination = tmp_path / "copy"
    destination.mkdir(exist_ok=True)

    run_from_uijson(
        uijson_path,
        destination=destination,
        monitoring_directory=monitoring_directory,
        relatives_only=False,
    )

    # test destination
    with Workspace(destination / "original.geoh5") as workspace:
        assert isinstance(workspace.get_entity("mean_xyz")[0], Data)

    ui_json_file = destination / "original.ui.json"
    with open(ui_json_file, encoding="utf-8") as file:
        ui_json_file = file.read()
        ui_json = json.loads(ui_json_file)

    assert ui_json["geoh5"].endswith("original.geoh5")
    assert ui_json["monitoring_directory"].endswith(str(monitoring_directory))

    # test monitoring directory
    monitor_geoh5 = next(iter(monitoring_directory.glob("*.geoh5")))
    with Workspace(monitoring_directory / monitor_geoh5) as workspace:
        assert isinstance(workspace.get_entity("mean_xyz")[0], Data)


def test_run_from_out_group(tmp_path, uijson_path):

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


def test_run_from_out_group_no_destination(tmp_path, uijson_path):
    run_from_outgroup_name(tmp_path / "original.geoh5", "uijson_test")

    # test destination
    with Workspace(tmp_path / "original.geoh5") as workspace:
        assert isinstance(workspace.get_entity("mean_xyz")[0], Data)


def test_out_group_errors(tmp_path, uijson_path):

    with Workspace(tmp_path / "original.geoh5") as workspace:
        # create an empty uijson group
        empty_group = UIJsonGroup.create(workspace, name="empty")

        with pytest.raises(ValueError, match="UIJsonGroup must have options"):
            run_uijson_group(empty_group)

        with pytest.raises(FileNotFoundError, match="Workspace file"):
            run_from_outgroup_name(tmp_path / "non_existent.geoh5", "uijson_test")

        with pytest.raises(TypeError, match="Entity with name"):
            run_from_outgroup_name(
                tmp_path / "original.geoh5",
                "inexistent",  # type: ignore
            )


def test_utils_errors(tmp_path, uijson_path):

    with pytest.raises(ValueError, match=r"Invalid ui\.json file"):
        load_ui_json_as_dict(123)  # type: ignore

    with pytest.raises(FileExistsError, match="File "):
        get_new_workspace_path("original.geoh5", tmp_path)
