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

import numpy as np
from geoh5py import Workspace
from geoh5py.objects import Curve, Grid2D, Points

from geoapps_utils.utils.locations import (
    gaussian,
    get_locations,
    get_overlapping_limits,
    map_indices_to_coordinates,
    mask_large_connections,
    mask_under_horizon,
)
from geoapps_utils.utils.transformations import rotate_points, z_rotation_matrix


def test_gaussian():
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = gaussian(x_grid, y_grid, 10, 5)
    assert np.isclose(z_grid.max(), 10, rtol=1e-3)


def test_mask_large_connections(tmp_path):
    with Workspace(tmp_path / "test.geoh5") as ws:
        x = np.linspace(0, 100, 11)
        y = np.linspace(0, 300, 4)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.zeros_like(x_grid)
        vertices = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
        crv = Curve.create(ws, name="test_curve", vertices=vertices)
        mask = mask_large_connections(crv, distance_threshold=50.0)
        assert len(mask) == 3


def test_rotate_points():
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 3, 4]])
    validation = rotate_points(
        points,
        (0.0, 0.0, 0.0),
        [z_rotation_matrix(90), z_rotation_matrix(np.array([-90] * 4))],
    )
    assert np.allclose(points, validation)


def test_mask_under_horizon():
    points = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [10, 10, 0]])
    horizon = np.array(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 1], [-0.5, 0.5, np.nan], [-1, 1, 0], [0, 0, 0]]
    )
    mask = mask_under_horizon(points, horizon)
    assert np.all(mask == np.array([True, False, False, True]))


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


def test_map_indices_to_coordinates():
    workspace = Workspace()
    n_x, n_y = 10, 15
    u_size, v_size = 20.0, 30.0
    grid = Grid2D.create(
        workspace,
        origin=[0, 0, 0],
        u_cell_size=u_size,
        v_cell_size=v_size,
        u_count=n_x,
        v_count=n_y,
        name="test_grid",
        allow_move=False,
    )
    y, x = np.meshgrid(
        np.arange(n_y) * v_size + v_size / 2, np.arange(n_x) * u_size + u_size / 2
    )

    ind_x = np.random.randint(0, n_x, 20)
    ind_y = np.random.randint(0, n_y, 20)

    indices = np.c_[ind_x, ind_y]

    np.testing.assert_array_equal(
        map_indices_to_coordinates(grid, indices),
        np.c_[
            x[indices[:, 0], indices[:, 1]].flatten(order="F"),
            y[indices[:, 0], indices[:, 1]].flatten(order="F"),
            np.zeros(20),
        ],
    )


def test_overlapping_limits():
    # Width too large
    limits = get_overlapping_limits(2, 5, overlap=0.25)
    assert limits == [[0, 2]]

    # Multiple overlap
    width = np.random.randint(2, 10)
    size = np.random.randint(2 * width, 10 * width)
    overlap = 0.25
    limits = get_overlapping_limits(size, width, overlap=overlap)

    assert all((lim[1] - lim[0]) == width for lim in limits)
    assert all(
        width / (limits[ind][1] - limits[ind + 1][0]) >= (1 + overlap)
        for ind in range(len(limits) - 1)
    )

    # No overlap
    overlap = 0
    limits = get_overlapping_limits(width * 4, width, overlap=overlap)

    assert limits[0][1] == limits[1][0]
