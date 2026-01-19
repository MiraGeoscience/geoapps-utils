# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2023-2026 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import numpy as np
from geoh5py import Workspace
from geoh5py.objects import BlockModel

from geoapps_utils.modelling.plates import PlateModel, inside_plate, make_plate


def test_inside_plate(tmp_path):
    with Workspace(tmp_path / "test.geoh5") as workspace:
        grid = BlockModel.create(
            workspace,
            name="test_block_model",
            u_cell_delimiters=np.linspace(-10, 10, 41),
            v_cell_delimiters=np.linspace(-10, 10, 41),
            z_cell_delimiters=np.linspace(-10, 10, 41),
            origin=np.r_[0, 0, 0],
        )

        strike_length = 15
        dip_length = 7
        width = 2
        mask = inside_plate(
            grid.centroids,
            plate=PlateModel(
                strike_length=strike_length,
                dip_length=dip_length,
                width=width,
                origin=(1.0, 0.0, 0.0),
            ),
        )
        model = np.zeros(len(grid.centroids))
        model[mask] = 2.0

        validation_mask = (
            (grid.centroids[:, 0] >= ((-strike_length / 2) + 1))
            & (grid.centroids[:, 0] <= ((strike_length / 2) + 1))
            & (grid.centroids[:, 1] >= -dip_length / 2)
            & (grid.centroids[:, 1] <= dip_length / 2)
            & (grid.centroids[:, 2] >= -width / 2)
            & (grid.centroids[:, 2] <= width / 2)
        )
        assert np.all(model[validation_mask] == 2.0)
        assert np.all(model[~validation_mask] == 0.0)


def test_make_plate(tmp_path):
    with Workspace(tmp_path / "test.geoh5") as workspace:
        grid = BlockModel.create(
            workspace,
            name="test_block_model",
            u_cell_delimiters=np.linspace(-10, 10, 41),
            v_cell_delimiters=np.linspace(-10, 10, 41),
            z_cell_delimiters=np.linspace(-10, 10, 41),
            origin=np.r_[0, 0, 0],
        )

        strike_length = 15
        dip_length = 7
        width = 2
        direction = 90
        dip = 0
        model = make_plate(
            grid.centroids,
            plate=PlateModel(
                strike_length=strike_length,
                dip_length=dip_length,
                width=width,
                direction=direction,
                dip=dip,
            ),
            background=0.0,
        )
        grid.add_data({"plate model 1": {"values": model}})

        mask = (
            (grid.centroids[:, 0] >= (-dip_length / 2))
            & (grid.centroids[:, 0] <= (dip_length / 2))
            & (grid.centroids[:, 1] >= (-strike_length / 2))
            & (grid.centroids[:, 1] <= (strike_length / 2))
            & (grid.centroids[:, 2] >= (-width / 2))
            & (grid.centroids[:, 2] <= (width / 2))
        )
        assert np.all(model[mask] == 1.0)

        direction = 0
        dip = 90
        model = make_plate(
            grid.centroids,
            plate=PlateModel(
                strike_length=strike_length,
                dip_length=dip_length,
                width=width,
                direction=direction,
                dip=dip,
            ),
            background=0.0,
        )
        grid.add_data({"plate model 2": {"values": model}})

        mask = (
            (grid.centroids[:, 0] >= (-strike_length / 2))
            & (grid.centroids[:, 0] <= (strike_length / 2))
            & (grid.centroids[:, 1] >= (-width / 2))
            & (grid.centroids[:, 1] <= (width / 2))
            & (grid.centroids[:, 2] >= (-dip_length / 2))
            & (grid.centroids[:, 2] <= (dip_length / 2))
        )
    assert np.all(model[mask] == 1.0)


def test_make_plate_multiple(tmp_path):
    with Workspace(tmp_path / "test.geoh5") as workspace:
        grid = BlockModel.create(
            workspace,
            name="test_block_model",
            u_cell_delimiters=np.linspace(-10, 10, 41),
            v_cell_delimiters=np.linspace(-10, 10, 41),
            z_cell_delimiters=np.linspace(-10, 10, 41),
            origin=np.r_[0, 0, 0],
        )

        strike_length = 15
        dip_length = 7
        width = 2
        direction = 90
        dip = 90
        plate = PlateModel(
            strike_length=strike_length,
            dip_length=dip_length,
            width=width,
            direction=direction,
            dip=dip,
            origin=(-1, 0, 0),
        )
        model = make_plate(
            grid.centroids,
            plate=plate,
            background=0.0,
        )
        plate.origin = (1, 0, 0)
        model = make_plate(grid.centroids, plate=plate, background=model)
        grid.add_data({"plate model": {"values": model}})

        mask = (
            (grid.centroids[:, 0] >= (-width))
            & (grid.centroids[:, 0] <= (width))
            & (grid.centroids[:, 1] >= (-strike_length / 2))
            & (grid.centroids[:, 1] <= (strike_length / 2))
            & (grid.centroids[:, 2] >= (-dip_length / 2))
            & (grid.centroids[:, 2] <= (dip_length / 2))
        )
        assert np.all(model[mask] == 1.0)
