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
from geoh5py.groups import UIJsonGroup
from geoh5py.objects import Points
from geoh5py.ui_json.input_file import InputFile

from geoapps_utils.run import (
    fetch_driver_class,
    load_ui_json,
    load_uijson_as_group,
    run_uijson_file,
    run_uijson_group,
)

from .dummy_driver_test import TestOptions


# pylint: disable=duplicate-code
def test_base_driver(tmp_path):
    workspace = Workspace.create(tmp_path / f"{__name__}.geoh5")

    points = Points.create(workspace)

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
        "out_group": None,
    }

    params = TestOptions.build(test_params)
    params._input_file = InputFile(ui_json=ui_json)  # pylint: disable=protected-access
    uijson_path = params.write_ui_json(path=tmp_path)
    run_uijson_file(uijson_path, tmp_path)


def test_load_fetch_uijson_errors(tmp_path):
    with pytest.raises(ValueError, match="Invalid ui.json file: "):
        load_ui_json(123)  # type: ignore

    uijson = {
        "title": "empty",
    }

    ui_json_path_1 = tmp_path / "temp1.ui.json"
    with open(ui_json_path_1, "w", encoding="utf-8") as file:
        json.dump(uijson, file)

    with pytest.raises(KeyError, match="'run_command' in ui.json must be a string"):
        fetch_driver_class(ui_json_path_1)

    uijson["run_command"] = "geoapps_utils.driver"

    ui_json_path_2 = tmp_path / "temp2.ui.json"
    with open(ui_json_path_2, "w", encoding="utf-8") as file:
        json.dump(uijson, file)

    with pytest.raises(KeyError, match="ui.json file"):
        load_uijson_as_group(ui_json_path_2)

    with pytest.raises(TypeError, match="Input 'out_group' must be"):
        run_uijson_group("bidon", path=tmp_path)  # type: ignore

    h5file = tmp_path / "test.geoh5"
    workspace = Workspace.create(h5file)
    ui_json_group = UIJsonGroup.create(workspace, name="test")

    with pytest.raises(ValueError, match="UIJsonGroup must have options"):
        run_uijson_group(ui_json_group, path=tmp_path)

    ui_json_group.options = uijson

    with pytest.raises(FileExistsError, match="File "):
        run_uijson_group(ui_json_group, path=tmp_path)
