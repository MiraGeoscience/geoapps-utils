#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

import numpy as np
from geoh5py import Workspace
from geoh5py.data import Data
from geoh5py.objects import Grid2D, Points
from geoh5py.objects.grid_object import GridObject


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

    if not isinstance(entity_obj, (Points, GridObject, Data)):
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


def filter_xy(
    x: np.array,
    y: np.array,
    distance: float | None = None,
    window: dict | None = None,
    angle: float | None = None,
    mask: np.ndarray | None = None,
) -> np.array:
    """
    Window and down-sample locations based on distance and window parameters.

    :param x: Easting coordinates, as vector or meshgrid-like array
    :param y: Northing coordinates, as vector or meshgrid-like array
    :param distance: Desired coordinate spacing.
    :param window: Window parameters describing a domain of interest.
        Must contain the following keys and values:
        window = {
            "center": [X: float, Y: float],
            "size": [width: float, height: float]
        }
        May also contain an "azimuth" in the case of rotated x and y.
    :param angle: Angle through which the locations must be rotated
        to take on a east-west, north-south orientation.  Supersedes
        the 'azimuth' key/value pair in the window dictionary if it
        exists.
    :param mask: Boolean mask to be combined with filter_xy masks via
        logical 'and' operation.

    :return mask: Boolean mask to be applied input arrays x and y.
    """

    if mask is None:
        mask = np.ones_like(x, dtype=bool)

    azim = None
    if angle is not None:
        azim = angle
    elif window is not None:
        if "azimuth" in window:
            azim = window["azimuth"]

    is_rotated = False if (azim is None) | (azim == 0) else True
    if is_rotated:
        xy_locs = rotate_xyz(np.c_[x.ravel(), y.ravel()], window["center"], azim)
        xr = xy_locs[:, 0].reshape(x.shape)
        yr = xy_locs[:, 1].reshape(y.shape)

    if window is not None:
        if is_rotated:
            mask, _, _ = window_xy(xr, yr, window, mask=mask)
        else:
            mask, _, _ = window_xy(x, y, window, mask=mask)

    if distance not in [None, 0]:
        is_grid = False
        if x.ndim > 1:
            if is_rotated:
                u_diff = np.unique(np.round(np.diff(xr, axis=1), 8))
                v_diff = np.unique(np.round(np.diff(yr, axis=0), 8))
            else:
                u_diff = np.unique(np.round(np.diff(x, axis=1), 8))
                v_diff = np.unique(np.round(np.diff(y, axis=0), 8))

            is_grid = (len(u_diff) == 1) & (len(v_diff) == 1)

        if is_grid:
            mask, _, _ = downsample_grid(x, y, distance, mask=mask)
        else:
            mask, _, _ = downsample_xy(x, y, distance, mask=mask)

    return mask


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
    left = np.linspace(0, size - width, n_tiles)
    limits = np.c_[left, left + width].astype(int)

    return limits.tolist()
