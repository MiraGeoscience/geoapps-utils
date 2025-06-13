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

import logging
from copy import deepcopy

import pytest
from geoh5py import Workspace
from geoh5py.ui_json.constants import default_ui_json as base_ui_json

from geoapps_utils.base import Options
from geoapps_utils.driver.data import BaseData
from geoapps_utils.driver.driver import BaseDriver
from geoapps_utils.driver.params import BaseParams


class TestParams(BaseParams):
    _default_ui_json = deepcopy(base_ui_json)

    def __init__(self, input_file=None, **kwargs):
        super().__init__(input_file=input_file, **kwargs)


def test_base_driver(tmp_path):
    workspace = Workspace.create(tmp_path / "test_workspace.geoh5")
    # Create params
    test_params = {
        "monitoring_directory": None,
        "workspace_geoh5": None,
        "geoh5": workspace,
        "run_command": None,
        "title": "test_title",
        "conda_environment": None,
        "conda_environment_boolean": False,
        "generate_sweep": False,
        "workspace": None,
        "run_command_boolean": False,
    }

    class TestDriver(BaseDriver):
        _params_class = TestParams

        def __init__(self, params: TestParams):
            super().__init__(params)

        def run(self):
            pass

    params = TestParams(**test_params)
    params.write_input_file(path=tmp_path, name="test_ifile.ui.json")

    # Create driver
    driver = TestDriver(params)
    driver.start(tmp_path / "test_ifile.ui.json")


def test_params_errors():
    with pytest.raises(TypeError, match="'input_data' must be "):
        BaseParams.build(input_data="bidon")  # type: ignore


def test_old_base_driver(caplog):
    ws = Workspace()
    params = Options(geoh5=ws)

    class TestDriver(BaseDriver):
        def run(self):
            pass

    with caplog.at_level(logging.WARNING):
        TestDriver(params)

    assert "removed in future release" in caplog.text


def test_old_base_options(caplog):
    ws = Workspace()

    with caplog.at_level(logging.WARNING):
        BaseData(geoh5=ws)

    assert "removed in future release" in caplog.text
