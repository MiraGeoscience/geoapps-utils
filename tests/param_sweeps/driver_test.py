# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2023-2026 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import itertools
import json
from pathlib import Path

import numpy as np
from geoh5py.objects import Points
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace

from geoapps_utils.base import Driver, Options
from geoapps_utils.param_sweeps.driver import SweepDriver, SweepParams
from geoapps_utils.param_sweeps.generate import generate


class TestDriver(Driver):
    _params_class = Options

    def __init__(self, params: Options):
        super().__init__(params)

    def run(self):
        pass


def test_params(tmp_path: Path):
    file = tmp_path / f"{__name__}.ui.geoh5"
    workspace = Workspace.create(file)
    options = Options(geoh5=workspace)
    test = options.serialize()
    test.update(
        {
            "param1_start": {"label": "param1 start", "value": 1},
            "param1_end": {"label": "param1 end", "value": 2},
            "param1_n": {"label": "param1 n samples", "value": 2},
            "worker_uijson": "worker.ui.json",
        }
    )
    ifile = InputFile(ui_json=test)
    params = SweepParams.build(ifile)

    assert Path(params.worker_uijson).name == "worker.ui.json"
    worker_params = params.worker_parameters()
    assert len(worker_params) == 1
    assert worker_params[0] == "param1"
    psets = params.parameter_sets()
    assert len(psets) == 1
    assert "param1" in psets
    assert psets["param1"] == [1, 2]


def test_uuid_from_params():
    test = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
    iterations = list(itertools.product(*test.values()))
    for iteration in iterations:
        trial_uuid = SweepDriver.uuid_from_params(iteration)
        assert trial_uuid == SweepDriver.uuid_from_params(iteration), (
            "method is not deterministic"
        )


def test_sweep(tmp_path: Path):  # pylint: disable=R0914
    uijson_path = tmp_path / "test.ui.json"
    sweep_path = tmp_path / "test_sweep.ui.json"

    with Workspace.create(tmp_path / f"{__name__}.ui.geoh5") as workspace:
        locs = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        pts = Points.create(workspace, name="data", vertices=locs)
        dat = pts.add_data({"initial": {"values": np.ones(4, dtype=np.int32)}})
        options = Options(geoh5=workspace)
        ui_json = options.serialize()
        ui_json.update(
            {
                "run_command": "tests.param_sweeps.driver_test",
                "data_object": {
                    "label": "Object",
                    "meshType": "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                    "value": pts,
                },
                "data": {
                    "association": "Vertex",
                    "dataType": "Integer",
                    "label": "data",
                    "parent": "data_object",
                    "value": dat,
                },
                "param": {"label": "Add value", "value": 1},
            }
        )
        ifile = InputFile(ui_json=ui_json)
        ifile.write_ui_json("test.ui.json", path=tmp_path)

    generate(str(uijson_path), parameters=["param"])

    with open(sweep_path, encoding="utf-8") as file:
        uijson = json.load(file)

    uijson["param_end"]["value"] = 2
    uijson["param_end"]["enabled"] = True
    uijson["param_n"]["value"] = 2
    uijson["param_n"]["enabled"] = True
    uijson["worker_uijson"] = str(tmp_path / "test.ui.json")

    with open(sweep_path, "w", encoding="utf-8") as file:
        json.dump(uijson, file, indent=4)

    SweepDriver.start(sweep_path, mode="r")

    with open(tmp_path / "lookup.json", encoding="utf-8") as file:
        lookup = json.load(file)

    assert all((tmp_path / f"{k}.ui.geoh5").is_file() for k in lookup)
    assert all((tmp_path / f"{k}.ui.json").is_file() for k in lookup)
    assert len(lookup.values()) == 2
    assert all(k["param"] in [1, 2] for k in lookup.values())

    for file_root in lookup:
        file_ws = Workspace(tmp_path / f"{file_root}.ui.geoh5")
        data = file_ws.get_entity("data")[0]
        assert isinstance(data, Points)
        assert any("initial" in k.name for k in data.children)
