# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2026 Mira Geoscience Ltd.                                          '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from pathlib import Path
from uuid import UUID

import pytest
from geoh5py import Workspace
from geoh5py.groups import UIJsonGroup
from geoh5py.objects import Points
from geoh5py.ui_json import UIJson


@pytest.fixture
def uijson_path(tmp_path) -> Path:
    with Workspace.create(tmp_path / "original.geoh5") as workspace:
        points = Points.create(workspace)

        out_group = UIJsonGroup.create(workspace, name="uijson_test")
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

        uijson_class = UIJson.infer(**ui_json)
        uijson = uijson_class(**ui_json)
        uijson_path = tmp_path / "original.ui.json"
        uijson.write(uijson_path)

    return uijson_path
