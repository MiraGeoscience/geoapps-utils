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

from uuid import UUID

import numpy as np
from geoh5py import Workspace
from geoh5py.data import Data
from geoh5py.objects import Grid2D, Points
from geoh5py.objects.grid_object import GridObject
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree


def mask_under_horizon(locations: np.ndarray, horizon: np.ndarray) -> np.ndarray:
    """
    Mask locations under a horizon.

    :param locations: A 3D distribution of x, y, z points data as an array
        of shape(*, 3).
    :param horizon: A quasi-2D distribution of x, y, z points data as an
        array of shape(*, 3) that forms a rough plane that intersects the
        provided locations 3D point cloud.

    :returns: A boolean array of shape(*, 1) where True values represent points
        in the locations array that lie below the triangulated horizon.
    """

    delaunay_2d = Delaunay(horizon[:, :-1])
    z_interpolate = LinearNDInterpolator(delaunay_2d, horizon[:, -1])
    z_locations = z_interpolate(locations[:, :2])

    outside = np.isnan(z_locations)
    if any(outside):
        tree = cKDTree(horizon)
        _, nearest = tree.query(locations[outside, :])
        z_locations[outside] = horizon[nearest, -1]

    below_horizon = locations[:, -1] < z_locations

    return below_horizon


def get_locations(workspace: Workspace, entity: UUID | Points | GridObject | Data):
    """
    Returns entity's centroids or vertices.

    If no location data is found on the provided entity, the method will
    attempt to call itself on its parent.

    :param workspace: Geoh5py Workspace entity.
    :param entity: Object or uuid of entity containing centroid or
        vertex location data.

    :return: Array shape(*, 3) of x, y, z location data

    """
    if isinstance(entity, UUID):
        entity_obj = workspace.get_entity(entity)[0]
    else:
        entity_obj = entity

    if not isinstance(entity_obj, Points | GridObject | Data):
        raise TypeError(
            f"Entity must be of type Points, GridObject or Data, {type(entity_obj)} provided."
        )

    if isinstance(entity_obj, Points):
        locations = entity_obj.vertices
    elif isinstance(entity_obj, GridObject):
        locations = entity_obj.centroids
    else:
        locations = get_locations(workspace, entity_obj.parent)

    return locations


def map_indices_to_coordinates(grid: Grid2D, indices: np.ndarray) -> np.ndarray:
    """
    Map indices to coordinates.

    :param grid: Grid2D object.
    :param indices: Indices (i, j) of grid cells.
    """

    if grid.centroids is None or grid.shape is None:
        raise ValueError("Grid2D object must have centroids.")

    x = grid.centroids[:, 0].reshape(grid.shape, order="F")
    y = grid.centroids[:, 1].reshape(grid.shape, order="F")
    z = grid.centroids[:, 2].reshape(grid.shape, order="F")

    return np.c_[
        x[indices[:, 0], indices[:, 1]],
        y[indices[:, 0], indices[:, 1]],
        z[indices[:, 0], indices[:, 1]],
    ]


def get_overlapping_limits(size: int, width: int, overlap: float = 0.25) -> list:
    """
    Get the limits of overlapping tiles.

    :param size: Number of cells along the axis.
    :param width: Size of the tile.
    :param overlap: Overlap factor between tiles [default=0.25].

    :returns: List of limits.
    """
    if size <= width:
        return [[0, int(size)]]

    n_tiles = int(np.ceil((1 + overlap) * size / width))

    def left_limits(n_tiles):
        left = np.linspace(0, size - width, n_tiles)
        return np.c_[left, left + width].astype(int)

    limits = left_limits(n_tiles)

    while np.any((limits[:-1, 1] - limits[1:, 0]) / width < overlap):
        n_tiles += 1
        limits = left_limits(n_tiles)

    return limits.tolist()
