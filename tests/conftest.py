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
from typing import ClassVar
from uuid import UUID

import pytest
from geoh5py import Workspace
from geoh5py.groups import UIJsonGroup
from geoh5py.objects import Points
from geoh5py.ui_json import UIJson
from pydantic import BaseModel, ConfigDict

from geoapps_utils import assets_path
from geoapps_utils.base import Options
from geoapps_utils.driver.driver import BaseDriver


class NestedModel(BaseModel):
    """
    Mock nested model
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Points | None = None


class TestOptions(Options):
    """
    Mock nested options
    """

    # todo: warning the base driver does not have a client attribute
    default_ui_json: ClassVar[Path] = assets_path() / "uijson/base.ui.json"
    nested_model: NestedModel


class TestOptionsDriver(BaseDriver):
    _params_class = TestOptions

    def __init__(self, params: TestOptions):
        super().__init__(params)

    def run(self):
        """
        Add a adata to the point to ensure something happens.
        """
        new_data = self.params.nested_model.client.vertices
        new_data = new_data.mean(axis=0)
        self.params.nested_model.client.add_data(
            {
                "mean_xyz": {
                    "value": new_data,
                }
            }
        )

        return self.params.nested_model.client


@pytest.fixture
def uijson_path(tmp_path) -> Path:
    with Workspace.create(tmp_path / "original.geoh5") as workspace:
        points = Points.create(workspace)

        out_group = UIJsonGroup.create(workspace, name="uijson_test")
        ui_json = {
            "version": "0.0.0",
            "title": "test_title",
            "conda_environment": "myenv",
            "run_command": "tests.conftest",
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
        out_path = tmp_path / "original.ui.json"
        uijson.write(out_path)

    return out_path
