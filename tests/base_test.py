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
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from geoh5py import Workspace
from geoh5py.objects import Points
from geoh5py.ui_json import InputFile, UIJson

from geoapps_utils.base import Options, get_logger
from geoapps_utils.driver.data import BaseData
from geoapps_utils.driver.driver import BaseDriver, Driver
from geoapps_utils.run import fetch_driver_class
from geoapps_utils.utils.importing import GeoAppsError

from .conftest import NestedModel, TestOptions, TestOptionsDriver


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


class TestNoDefaultOptions(Options):
    """
    Mock nested options
    """

    # todo: warning the base driver does not have a client attribute
    default_ui_json: ClassVar[Path] = Path("uijson/something.ui.json")
    nested_model: NestedModel


def test_base_options(tmp_path):
    workspace = Workspace.create(tmp_path / f"{__name__}.geoh5")
    # Create params
    pts = Points.create(workspace, vertices=np.random.randn(10, 3))
    options = TestOptions.build({"geoh5": workspace, "client": pts})

    with pytest.raises(ValueError, match="No output group"):
        options.update_out_group_options()

    driver = TestOptionsDriver(options)

    assert TestOptionsDriver.get_default_ui_json_path().exists()  # type: ignore

    assert isinstance(driver.params, TestOptions)
    assert driver.params_class == TestOptions
    assert isinstance(driver.workspace, Workspace)
    assert driver.out_group is None

    demoted = options.serialize(mode="json")
    assert demoted["client"] == str(pts.uid)

    # Write the options as file attached
    driver.update_monitoring_directory(pts)

    assert len(pts.children) == 1
    file_data = pts.children[0]
    assert file_data.name == "temp.ui.json"

    json_dict = json.loads(file_data.file_bytes.decode())
    assert json_dict.get("client", None) == str(pts.uid)


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
    with pytest.raises(KeyError, match=r"'run_command' in ui\.json must be a string"):
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


class NotOptionsDriver(Driver):
    _params_class = Points  # type: ignore

    def run(self):
        pass


def test_warning_options(tmp_path):

    with pytest.raises(ValueError, match=r"does not have a default ui.json"):
        TestNoDefaultOptions.get_default_ui_json()

    options = TestOptions.model_construct(
        geoh5=Workspace.create(tmp_path / "test.geoh5"), nested_model=NestedModel()
    )
    with pytest.warns(DeprecationWarning, match=r"InputFile property is deprecated"):
        ui_json = options.input_file

    assert isinstance(ui_json, UIJson)

    with pytest.raises(TypeError, match=r"Input data must be a dictionary"):
        TestOptions.build(123)  # type: ignore

    ws = Workspace()
    pts = Points.create(ws, vertices=np.random.randn(10, 3))
    driver = NotOptionsDriver(pts)

    with pytest.raises(ValueError, match=r"does not have a default ui.json"):
        driver.get_default_ui_json()

    ifile = InputFile()
    with pytest.raises(
        GeoAppsError, match=r"The application needs a valid 'ui_json' file"
    ):
        TestOptions.build(ifile)
