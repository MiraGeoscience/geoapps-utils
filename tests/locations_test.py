#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np
from geoh5py import Workspace
from geoh5py.objects import Grid2D, Points

from geoapps_utils.locations import get_locations


def test_get_locations_centroids():
    workspace = Workspace()
    n_x, n_y = 10, 15
    grid = Grid2D.create(
        workspace,
        origin=[0, 0, 0],
        u_cell_size=20.0,
        v_cell_size=30.0,
        u_count=n_x,
        v_count=n_y,
        name="test_grid",
        allow_move=False,
    )
    # Test get_locations with centroids
    base_locs = get_locations(workspace, grid)

    assert base_locs.shape == (n_x * n_y, 3)

    # Test get_locations with a child of the grid
    test_data = grid.add_data({"test_data": {"values": np.ones(10 * 15)}})
    data_locs = get_locations(workspace, test_data)

    np.testing.assert_array_equal(base_locs, data_locs)

    # Test from uuid
    base_locs_from_uuid = get_locations(workspace, grid.uid)
    assert np.all(base_locs_from_uuid == base_locs)


def test_get_locations_vertices():
    workspace = Workspace()
    # Test get_locations with vertices
    vertices = np.random.rand(10, 3)
    points = Points.create(
        workspace,
        name="test_points",
        vertices=vertices,
    )
    assert np.all(get_locations(workspace, points) == vertices)
