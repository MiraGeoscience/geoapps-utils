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

from uuid import UUID

from geoh5py import Workspace
from geoh5py.objects import Points
from geoh5py.ui_json.input_file import InputFile

from geoapps_utils.run import run_uijson_file

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
        "run_command": "tests.dummy_driver",
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
        "run_command": "tests.dummy_driver",
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
