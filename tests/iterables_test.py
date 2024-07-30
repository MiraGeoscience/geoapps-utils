#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from geoh5py import Workspace
from geoh5py.objects import Grid2D

from geoapps_utils.iterables import (
    find_value,
    sorted_alphanumeric_list,
    sorted_children_dict,
)


def test_find_value_in_list():
    labels = ["inversion_01_model", "inversion_01_data", "inversion_02_model"]
    assert find_value(labels, ["data"]) == "inversion_01_data"
    assert find_value(labels, ["inversion", "02"]) == "inversion_02_model"
    assert find_value(labels, ["inversion"]) == "inversion_02_model"
    assert find_value(labels, ["lskdfjsd"]) is None


def test_find_value_in_pairs():
    labels = [["inversion_01_model", 1], ["inversion_01_data", 2]]
    assert find_value(labels, ["model"]) == 1
    assert find_value(labels, ["data"]) == 2
    assert find_value(labels, ["lskdjf"]) is None


def test_sorted_alphanumeric_list():
    test = [
        "Iteration_3.2e-1_data",
        "Iteration_1_data",
        "Iteration_2_data",
        "Iteration_3_data",
        "Iteration_5.11_data",
        "Iteration_5.2_data",
        "Iteration_6_data",
        "Iteration_7_data",
        "Iteration_8e0_data",
        "Iteration_9.0_data",
        "Iteration_10_data",
        "Iteration_11_data",
        "Iteration_2_model",
        "Iteration_12_model",
        "interp_01",
        "interp_02",
        "interp_11",
        "iteration_2_model",
        "iteration_12_model",
        "topo",
        "uncert",
    ]

    sorted_list = sorted_alphanumeric_list(random.sample(test, len(test)))
    assert all(elem == tester for elem, tester in zip(sorted_list, test, strict=False))


def test_sorted_children_dict(tmp_path: Path):
    workspace = Workspace.create(tmp_path / "test.geoh5")
    n_x, n_y = 10, 15
    grid = Grid2D.create(
        workspace,
        origin=[10, 10, 10],
        u_cell_size=30.0,
        v_cell_size=20.0,
        u_count=n_x,
        v_count=n_y,
        name="test_grid",
        allow_move=False,
    )

    grid.add_data({"Iteration_10_data": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_1_data": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_5_data": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_3_data": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_2_data": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_4_data": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_9.0_data": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_8e0_data": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_11_data": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_6_data": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_7_data": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"interp_02": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"interp_01": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"interp_11": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"iteration_2_model": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"iteration_12_model": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_2_model": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"Iteration_12_model": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"topo": {"values": np.ones(grid.n_cells)}})
    grid.add_data({"uncert": {"values": np.ones(grid.n_cells)}})
    grid.add_data(
        {
            "wrong_data_type": {
                "type": "text",
                "values": np.array(["test" for _ in range(grid.n_cells)], dtype=object),
            }
        }
    )

    data_dict = sorted_children_dict(grid)
    assert data_dict is not None
    data_keys = list(data_dict.keys())
    for i, k in enumerate(data_keys):
        print(i, k)
    assert data_keys[0] == "Iteration_1_data"
    assert data_keys[1] == "Iteration_2_data"
    assert data_keys[7] == "Iteration_8e0_data"
    assert data_keys[8] == "Iteration_9.0_data"
    assert data_keys[11] == "Iteration_2_model"
    assert data_keys[12] == "Iteration_12_model"
    assert data_keys[16] == "iteration_2_model"
    assert data_keys[17] == "iteration_12_model"
    assert data_keys[-2] == "topo"
    assert data_keys[-1] == "uncert"
    assert "wrong_data_type" not in data_keys

    data_dict_from_uuid = sorted_children_dict(grid.uid, workspace)
    assert data_dict == data_dict_from_uuid

    assert sorted_children_dict(grid.uid, None) is None
