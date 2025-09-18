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

import json
import logging
from copy import deepcopy
from uuid import UUID

import numpy as np
import pytest
from geoh5py import Workspace
from geoh5py.groups import UIJsonGroup
from geoh5py.objects import Points
from geoh5py.ui_json import InputFile
from geoh5py.ui_json.templates import group_parameter, object_parameter

from geoapps_utils.base import Options, get_logger
from geoapps_utils.driver.data import BaseData
from geoapps_utils.driver.driver import BaseDriver, Driver
from geoapps_utils.driver.params import BaseParams
from geoapps_utils.run import fetch_driver_class

from .dummy_driver_test import (
    TestOptions,
    TestOptionsDriver,
    TestParams,
    TestParamsDriver,
)


TEST_DICT = {
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "geoh5": None,
    "run_command": None,
    "title": "test_title",
    "conda_environment": None,
    "conda_environment_boolean": False,
    "generate_sweep": False,
    "workspace": None,
    "run_command_boolean": False,
}


def test_base_driver(tmp_path):
    workspace = Workspace.create(tmp_path / f"{__name__}.geoh5")
    # Create params
    test_params = deepcopy(TEST_DICT)
    test_params["geoh5"] = str(workspace.h5file)

    params = TestParams(**test_params)
    params.write_input_file(path=tmp_path, name="test_ifile.ui.json")

    # Create driver
    with pytest.raises(TypeError, match="Parameters must be of type"):
        TestParamsDriver("not a params object")  # type: ignore

    driver = TestParamsDriver(params)
    driver.start(tmp_path / "test_ifile.ui.json")


def test_options_out(tmp_path):
    workspace = Workspace.create(tmp_path / f"{__name__}.geoh5")
    # Create params
    pts = Points.create(workspace, vertices=np.random.randn(10, 3))
    out_group = UIJsonGroup.create(workspace, name="Test Group")

    with pytest.raises(TypeError, match="Input data must be a dictionary"):
        TestOptions.build("not a dict")  # type: ignore

    # test creation of input file on the fly
    options = TestOptions.build({"geoh5": workspace, "client": pts})
    assert options._input_file is None  # pylint: disable=protected-access
    options.write_ui_json(tmp_path / "test_options.ui.json")
    assert isinstance(options._input_file, InputFile)  # pylint: disable=protected-access

    ui_json = deepcopy(TEST_DICT)
    ui_json["out_group"] = group_parameter(value=out_group)
    ui_json["geoh5"] = workspace
    ui_json["client"] = object_parameter(value=pts)
    ui_json["run_command"] = "geoapps_utils.driver.driver"

    ifile = InputFile(ui_json=ui_json)
    options = TestOptions.build(ifile)

    # Test updating out_group options
    assert len(out_group.options) == 0
    options.update_out_group_options()
    assert len(out_group.options) > 0
    assert UUID(out_group.options["out_group"]["value"]) == out_group.uid


def test_base_options(tmp_path):
    workspace = Workspace.create(tmp_path / f"{__name__}.geoh5")
    # Create params
    pts = Points.create(workspace, vertices=np.random.randn(10, 3))

    with pytest.raises(TypeError, match="Input data must be a dictionary"):
        TestOptions.build("not a dict")  # type: ignore

    options = TestOptions.build({"geoh5": workspace, "client": pts})

    with pytest.raises(ValueError, match="No output group"):
        options.update_out_group_options()

    driver = TestOptionsDriver(options)

    assert isinstance(driver.params, TestOptions)
    assert driver.params_class == TestOptions
    assert isinstance(driver.workspace, Workspace)
    assert driver.out_group is None

    demoted = options.serialize()
    assert demoted["client"] == "{" + str(pts.uid) + "}"

    # Write the options as file attached
    driver.update_monitoring_directory(pts)

    assert len(pts.children) == 1
    file_data = pts.children[0]
    assert file_data.name == "base.ui.json"

    json_dict = json.loads(file_data.file_bytes.decode())
    assert json_dict.get("client", None) == "{" + str(pts.uid) + "}"


def test_params_errors():
    with pytest.raises(TypeError, match="'input_data' must be "):
        BaseParams.build(input_data="bidon")  # type: ignore


def test_old_base_driver(caplog):
    ws = Workspace()
    params = Options(geoh5=ws)

    class TestDriver(BaseDriver):
        _params_class = Options

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


def test_fetch_driver(tmp_path):
    params = Options.model_construct()  # type: ignore
    dict_params = params.model_dump()
    with open(tmp_path / f"{__name__}.ui.json", "w", encoding="utf-8") as file:
        json.dump(params.model_dump(), file, indent=4)

    driver_class = fetch_driver_class(tmp_path / f"{__name__}.ui.json")

    assert driver_class is Driver

    # Repeat with bad run_command
    dict_params["run_command"] = "hello.world"
    with pytest.raises(ModuleNotFoundError, match="No module named 'hello'"):
        fetch_driver_class(dict_params)

    # Repeat with missing run_command
    del dict_params["run_command"]
    with pytest.raises(KeyError, match="'run_command' in ui.json must be a string"):
        fetch_driver_class(dict_params)

    # Repeat with missing driver in module
    dict_params["run_command"] = "geoapps_utils.utils.plotting"
    with pytest.raises(SystemExit, match="1"):
        fetch_driver_class(dict_params)


def test_logger(caplog):
    """
    Test that the logger is set up correctly.
    """
    logger = get_logger("my-app")
    with caplog.at_level("INFO"):
        logger.info("Test log message")

    assert "Test log message" in caplog.text
    assert "my-app" in caplog.text
    assert caplog.records[0].levelname == "INFO"
    assert caplog.records[0].name == "my-app"
