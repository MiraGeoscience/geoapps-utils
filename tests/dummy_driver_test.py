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

import sys
from copy import deepcopy
from pathlib import Path
from typing import ClassVar

from geoh5py.objects import Points
from geoh5py.ui_json.constants import default_ui_json as base_ui_json
from pydantic import BaseModel, ConfigDict

from geoapps_utils import assets_path
from geoapps_utils.base import Options
from geoapps_utils.driver.driver import BaseDriver
from geoapps_utils.driver.params import BaseParams


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


class TestParams(BaseParams):
    _default_ui_json = deepcopy(base_ui_json)

    def __init__(self, input_file=None, **kwargs):
        super().__init__(input_file=input_file, **kwargs)


class TestOptionsDriver(BaseDriver):
    _params_class = TestOptions

    def __init__(self, params: TestOptions):
        super().__init__(params)

    def run(self):
        pass


class TestParamsDriver(BaseDriver):
    _params_class = TestParams

    def __init__(self, params: TestParams):
        super().__init__(params)

    def run(self):
        pass


if __name__ == "__main__":
    TestOptionsDriver.start(sys.argv[1])
