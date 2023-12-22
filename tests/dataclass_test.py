#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import pytest
from geoh5py.workspace import Workspace
from pydantic import ValidationError

from geoapps_utils.driver.data import BaseData


def test_dataclass_valid_values(tmp_path):
    workspace = Workspace(tmp_path / "test.geoh5")

    valid_params = {
        "monitoring_directory": workspace.h5file,
        "workspace_geoh5": workspace.h5file,
        "geoh5": workspace.h5file,
        "run_command": "test.driver",
        "title": "test title",
        "conda_environment": "test_env",
        "conda_environment_boolean": True,
        "workspace": workspace,
        "run_command_boolean": False,
    }

    model = BaseData(**valid_params)
    output_params = {**model.model_dump()}

    for k, v in output_params.items():
        assert output_params[k] == v

    assert len(output_params) == len(valid_params)

    for k, v in valid_params.items():
        assert output_params[k] == v


def test_dataclass_invalid_values(tmp_path):
    workspace = Workspace(tmp_path / "test.geoh5")

    invalid_params = {
        "monitoring_directory": 5,
        "workspace_geoh5": workspace.h5file,
        "geoh5": False,
        "run_command": "test.driver",
        "title": None,
        "conda_environment": "test_env",
        "workspace": workspace,
        "run_command_boolean": False,
    }

    with pytest.raises(ValidationError) as e:
        BaseData(**invalid_params)
        assert len(e.errors()) == 6
        error_params = [error["loc"][0] for error in e.errors()]
        error_types = [error["type"] for error in e.errors()]
        for error_param in [
            "monitoring_directory",
            "geoh5",
            "title",
            "conda_environment_boolean",
        ]:
            assert error_param in error_params
        for error_type in ["string_type", "path_type", "missing"]:
            assert error_type in error_types
