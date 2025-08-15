# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2023-2025 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import json
from pathlib import Path

from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace

from geoapps_utils.base import Options
from geoapps_utils.param_sweeps.generate import generate, sweep_forms


def test_generate(tmp_path: Path):
    workspace = Workspace(tmp_path / f"{__name__}.ui.geoh5")

    options = Options(geoh5=workspace)
    ui_json = options.serialize()
    ui_json.update(
        {
            "param1": {"label": "param1", "value": 1},
            "param2": {"label": "param2", "value": 2.5},
        }
    )
    ifile = InputFile(ui_json=ui_json)
    path = tmp_path / "worker.ui.json"
    ifile.write_ui_json("worker.ui.json", path=tmp_path)

    generate(str(path))
    with open(path.parent / "worker_sweep.ui.json", encoding="utf8") as file:
        data = json.load(file)

    assert "param1_start" in data
    assert "param1_end" in data
    assert "param1_n" in data
    assert "param2_start" in data
    assert "param2_end" in data
    assert "param2_n" in data

    assert not data["param1_end"]["enabled"]
    assert not data["param1_n"]["enabled"]
    assert data["param1_n"]["dependency"] == "param1_end"
    assert not data["param2_end"]["enabled"]
    assert not data["param2_n"]["enabled"]
    assert data["param2_n"]["dependency"] == "param2_end"

    generate(str(path), parameters=["param1"])

    with open(path.parent / "worker_sweep.ui.json", encoding="utf8") as file:
        data = json.load(file)

    assert "param2_start" not in data
    assert "param2_end" not in data
    assert "param2_n" not in data


def test_sweep_forms():
    forms = sweep_forms("test", 1)
    params = ["test_start", "test_end", "test_n"]
    assert len(forms) == 3
    assert all(k in forms for k in params)
    assert all(forms[k]["group"] == "Test" for k in params)
    assert all(f["value"] == 1 for f in forms.values())
    assert forms["test_end"]["optional"]
    assert not forms["test_end"]["enabled"]
    assert forms["test_n"]["dependency"] == "test_end"
    assert forms["test_n"]["dependencyType"] == "enabled"
    assert not forms["test_n"]["enabled"]
