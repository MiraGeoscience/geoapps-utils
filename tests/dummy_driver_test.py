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

import sys
from pathlib import Path
from typing import ClassVar

from geoh5py.objects import Points
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

        self.update_monitoring_directory(self.params.nested_model.client)


if __name__ == "__main__":
    TestOptionsDriver.start(sys.argv[1])
