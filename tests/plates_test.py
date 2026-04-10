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

import numpy as np
import pytest
from geoh5py import Workspace
from geoh5py.objects import BlockModel

from geoapps_utils.modelling.plates import Plate, PlateModel, inside_plate, make_plate


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
                easting=1.0,
                northing=0.0,
                elevation=0.0,
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
                easting=0.0,
                northing=0.0,
                elevation=0.0,
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
                easting=0.0,
                northing=0.0,
                elevation=0.0,
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
            easting=-1.0,
            northing=0.0,
            elevation=0.0,
            direction=direction,
            dip=dip,
        )
        model = make_plate(
            grid.centroids,
            plate=plate,
            background=0.0,
        )
        plate.easting = 1.0
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


def test_plate_alias():
    plate = PlateModel(
        strike_length=15,
        dip_length=7,
        width=2,
        easting=0.0,
        northing=0.0,
        elevation=0.0,
        direction=90,
        dip=0,
    )
    assert plate.direction == 90
    assert "direction" in plate.model_dump()

    plate = PlateModel(
        strike_length=15,
        dip_length=7,
        width=2,
        easting=0.0,
        northing=0.0,
        elevation=0.0,
        dip_direction=90,
        dip=0,
    )
    assert plate.direction == 90
    assert "dip_direction" in plate.model_dump(by_alias=True)


def test_maxwell_plate_integration(tmp_path):

    plate = Plate(
        PlateModel(
            strike_length=100,
            dip_length=300,
            width=20,
            easting=100.0,
            northing=0.0,
            elevation=0.0,
            dip_direction=90,
            dip=45,
        )
    )
    with Workspace(tmp_path / "test.geoh5") as workspace:
        maxwell_plate = plate.to_maxwell_plate(workspace)
        assert maxwell_plate.geometry is not None
        maxwell_plate.geometry.rotation = 10.0

    with pytest.warns(UserWarning, match="Plunging plate"):
        plate = Plate.from_maxwell_plate(maxwell_plate)

    assert plate.params.strike_length == 100
    assert plate.params.dip_length == 300
    assert plate.params.width == 20
    assert plate.params.easting == 100
    assert plate.params.northing == 0
    assert plate.params.elevation == 0
    assert plate.params.direction == 90
    assert plate.params.dip == 45
