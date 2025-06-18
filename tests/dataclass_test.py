# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2023-2025 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#  pylint: disable=too-few-public-methods

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from geoh5py.groups import DrillholeGroup
from geoh5py.objects import Drillhole
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from pydantic import BaseModel, ValidationError

from geoapps_utils import assets_path
from geoapps_utils.base import Options
from geoapps_utils.utils.importing import DrillholeGroupValue


def get_params_dict(tmp_path):
    workspace = Workspace.create(tmp_path)
    param_dict = {
        "monitoring_directory": None,
        "geoh5": workspace,
        "run_command": "test.driver",
        "title": "test title",
        "conda_environment": "test_env",
        "out_group": None,
    }
    return param_dict


class TestOpts(BaseModel):
    opt1: str
    opt2: str = "default"
    opt3: str | None = None


class TestParams(BaseModel):
    type: str
    options: TestOpts


class TestModel(BaseModel):
    name: str
    value: float
    params: TestParams


def test_dataclass_valid_values(tmp_path):
    valid_parameters = get_params_dict(tmp_path / f"{__name__}.geoh5")
    model = Options(**valid_parameters)
    output_params = model.model_dump()
    assert len(output_params) == len(valid_parameters)

    for k, v in output_params.items():
        assert valid_parameters[k] == v


def test_dataclass_invalid_values(tmp_path):
    workspace = Workspace(tmp_path / "test.geoh5")

    invalid_params = {
        "monitoring_directory": 5,
        "workspace_geoh5": workspace.h5file,
        "geoh5": False,
        "run_command": "test.driver",
        "title": 123,
        "conda_environment": "test_env",
        "workspace": workspace.h5file,
    }
    try:
        Options(**invalid_params)
    except ValidationError as e:
        assert len(e.errors()) == 4  # type: ignore
        error_params = [error["loc"][0] for error in e.errors()]  # type: ignore
        error_types = [error["type"] for error in e.errors()]  # type: ignore
        for error_param in [
            "monitoring_directory",
            "geoh5",
            "title",
        ]:
            assert error_param in error_params
        for error_type in ["string_type", "path_type", "is_instance_of"]:
            assert error_type in error_types


def test_dataclass_input_file(tmp_path):
    valid_parameters = get_params_dict(tmp_path / f"{__name__}.geoh5")
    ifile = InputFile(ui_json=valid_parameters)
    model = Options.build(ifile)

    assert model.geoh5.h5file == tmp_path / f"{__name__}.geoh5"
    assert model.flatten() == valid_parameters
    assert model._input_file == ifile  # pylint: disable=protected-access


def test_pydantic_validates_nested_models():
    with pytest.raises(ValidationError):
        TestModel(
            name="test",
            value=1.0,
            params=TestParams(
                type="big",
                options=TestOpts(opt2="opt2", opt3="opt3"),  # type: ignore
            ),
        )

    with pytest.raises(ValidationError):
        TestModel(
            **{
                "name": "test",
                "value": 1.0,
                "params": {
                    "type": "big",
                    "options": {
                        "opt2": "opt2",
                        "opt3": "opt3",
                    },
                },
            },
        )


def test_collect_input_from_dict():
    test_data = {
        "name": "test",
        "value": 1.0,
        "type": "big",
        "opt1": "opt1",
        "opt2": "opt2",
        "opt3": "opt3",
    }

    data = Options.collect_input_from_dict(TestModel, test_data)  # type: ignore
    assert data["name"] == "test"
    assert data["value"] == 1.0
    assert data["params"]["type"] == "big"
    assert data["params"]["options"]["opt1"] == "opt1"
    assert data["params"]["options"]["opt2"] == "opt2"
    assert data["params"]["options"]["opt3"] == "opt3"


def test_missing_parameters(tmp_path):
    valid_parameters = get_params_dict(tmp_path / f"{__name__}.geoh5")
    test_data = {
        "name": "test",
        "type": "big",
        "opt1": "opt1",
        "opt2": "opt2",
        "opt3": "opt3",
    }
    kwargs = Options.collect_input_from_dict(TestModel, test_data)  # type: ignore
    with pytest.raises(ValidationError, match="value\n  Field required"):
        TestModel(**valid_parameters, **kwargs)

    test_data = {
        "name": "test",
        "value": 1.0,  # type: ignore
        "type": "big",
        "opt2": "opt2",
        "opt3": "opt3",
    }
    kwargs = Options.collect_input_from_dict(TestModel, test_data)  # type: ignore
    with pytest.raises(ValidationError, match="opt1\n  Field required"):
        TestModel(**valid_parameters, **kwargs)

    test_data = {
        "name": "test",
        "value": 1.0,  # type: ignore
        "type": "big",
        "opt1": "opt1",
        "opt3": "opt3",
    }
    kwargs = Options.collect_input_from_dict(TestModel, test_data)  # type: ignore
    model = TestModel(**valid_parameters, **kwargs)
    assert model.params.options.opt2 == "default"


def test_nested_model(tmp_path):
    class GroupOptions(BaseModel):
        group_type: str

    class GroupParams(BaseModel):
        value: str
        options: GroupOptions

    class NestedModel(Options):
        """
        Example of nested model
        """

        _name = "nested"
        group: GroupParams

    valid_params = get_params_dict(tmp_path / f"{__name__}.geoh5")
    valid_params["value"] = "test"
    valid_params["group_type"] = "multi"

    ifile = InputFile(ui_json=valid_params)
    model = NestedModel.build(ifile)

    assert isinstance(model.group, GroupParams)
    assert model.group.value == "test"
    assert model.flatten() == valid_params
    assert model.group.options.group_type == "multi"


def test_params_construction(tmp_path):
    params = Options(geoh5=Workspace(tmp_path / "test.geoh5"))
    assert Options.default_ui_json is None
    assert params.title == "Base Data"
    assert params.run_command == "geoapps_utils.base"
    assert str(params.geoh5.h5file) == str(tmp_path / "test.geoh5")


def test_base_data_write_ui_json(tmp_path):
    class TestData(Options):
        default_ui_json: ClassVar[Path | None] = assets_path() / "uijson/base.ui.json"

    params = TestData(geoh5=Workspace(tmp_path / "test.geoh5"))
    params.write_ui_json(tmp_path / "test.ui.json")
    assert (tmp_path / "test.ui.json").exists()

    ifile = InputFile.read_ui_json(
        assets_path() / "uijson/base.ui.json", validate=False
    )
    ifile.ui_json["my_param"] = "test it"
    ifile.data["my_param"] = "test it"
    ifile.data["geoh5"] = params.geoh5
    params2 = Options.build(ifile)
    params2.write_ui_json(tmp_path / "validation.ui.json")

    ifile = InputFile.read_ui_json(tmp_path / "validation.ui.json")
    assert ifile.data["my_param"] == "test it"

    ifile.data = None
    params3 = Options(geoh5=Workspace(tmp_path / "test.geoh5"), _input_file=ifile)

    assert isinstance(params3._create_input_file_from_attributes(), InputFile)  # pylint: disable=protected-access


def test_drillhole_groups(tmp_path):
    h5path = tmp_path / "test.geoh5"

    class MyData(Options):
        drillholes: DrillholeGroupValue

    with Workspace(h5path) as workspace:
        drillhole_group: DrillholeGroup = DrillholeGroup.create(
            workspace, name="drillhole_group_test"
        )

        for i in range(4):
            well = Drillhole.create(
                workspace,
                name=f"drillhole_test_{i}",
                default_collocation_distance=1e-5,
                parent=drillhole_group,
                collar=[10.0 * i, 10.0 * i, 10],
            )
            well.surveys = np.c_[
                np.linspace(0, 10, 4),
                np.ones(4) * 45.0,
                np.linspace(-89, -75, 4),
            ]

            # Create random from-to
            depth_ = np.sort(np.random.uniform(low=0.05, high=10, size=(40,))).reshape(
                (-1, 2)
            )

            # Add from-to data
            well.add_data(
                {
                    "interval_values": {
                        "values": np.random.randn(depth_.shape[0]),
                        "from-to": depth_.tolist(),
                    },
                    "interval_values_2": {
                        "values": np.random.randn(depth_.shape[0]),
                        "from-to": depth_.tolist(),
                    },
                },
                property_group="interval",
            )

        dh_g_params = {
            "geoh5": workspace,
            "title": "test_drillholes",
            "drillholes": {
                "label": "drillhole_group_test",
                "main": True,
                "multiSelect": False,
                "groupType": "825424FB-C2C6-4FEA-9F2B-6CD00023D393",
                "group_value": drillhole_group,
                "value": "interval_values",
            },
        }

        input_file = MyData(**dh_g_params)

        assert input_file.drillholes.group_value == drillhole_group
        assert input_file.drillholes.value == ["interval_values"]
