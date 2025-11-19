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

from logging import getLogger
from uuid import UUID

import numpy as np
from geoh5py import Workspace
from geoh5py.data import Data
from geoh5py.objects import CellObject, Grid2D, Points
from geoh5py.objects.grid_object import GridObject
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree


_logger = getLogger(__name__)


def gaussian(
    x: np.ndarray, y: np.ndarray, amplitude: float, width: float
) -> np.ndarray:
    """
    Gaussian function for 2D data.

    :param x: X-coordinates.
    :param y: Y-coordinates.
    :param amplitude: Amplitude of the Gaussian.
    :param width: Width parameter of the Gaussian.
    """

    return amplitude * np.exp(-0.5 * ((x / width) ** 2.0 + (y / width) ** 2.0))


def mask_large_connections(cell_object: CellObject, distance_threshold: float):
    """
    Trim connections in cell based objects.

    :param cell_object: Cell object containing segments with small vertex spacing
        along-line, but large spacing between segments.

    :return: Cleaned object without cells exceeding the distance threshold.
    """

    dist = np.linalg.norm(
        cell_object.vertices[cell_object.cells[:, 0], :]
        - cell_object.vertices[cell_object.cells[:, 1], :],
        axis=1,
    )

    return np.where(dist > distance_threshold)[0]


def topo_drape_elevation(
    locations: np.ndarray,
    topo: np.ndarray,
    method="linear",
    triangulation: np.ndarray | None = None,
) -> np.ndarray:
    """
    Get draped elevation at locations.

    Values are extrapolated to nearest neighbour if requested outside the
    convex hull of the input topography points.

    :param locations: n x 3 array of locations
    :param topo: n x 3 array of topography points
    :param method: Type of topography interpolation, either 'linear' or 'nearest'
    :param triangulation: Optional array describing triangulation

    :return: An array of z elevations for every input locations.
    """
    actives = ~np.any(np.isnan(topo), axis=1)
    unique_locs, inds = np.unique(
        locations[:, :-1].round(), axis=0, return_inverse=True
    )

    if method == "linear":
        tr = Triangulation(topo[:, 0], topo[:, 1], triangles=triangulation)
        z_interpolate = LinearTriInterpolator(tr, topo[:, 2])
        z_locations = z_interpolate(unique_locs[:, 0], unique_locs[:, 1]).data[inds]
    elif method == "nearest":
        z_interpolate = NearestNDInterpolator(topo[actives, :-1], topo[actives, -1])
        z_locations = z_interpolate(unique_locs)[inds]
    else:
        raise ValueError("Method must be 'linear', or 'nearest'")

    # Apply nearest neighbour if in extrapolation
    ind_nan = np.isnan(z_locations)
    if any(ind_nan):
        _logger.warning(
            "Locations found outside the convex hull of topography.\n"
            "Elevations will be extrapolated using a nearest neighbour."
        )
        tree = cKDTree(topo[actives, :])
        _, ind = tree.query(locations[ind_nan, :])
        z_locations[ind_nan] = topo[ind, -1]

    return np.c_[locations[:, :-1], z_locations]


def mask_under_horizon(
    locations: np.ndarray,
    horizon: np.ndarray,
    method="linear",
    triangulation: np.ndarray | None = None,
) -> np.ndarray:
    """
    Mask locations under a horizon.

    :param locations: A 3D distribution of x, y, z points data as an array
        of shape(*, 3).
    :param horizon: A quasi-2D distribution of x, y, z points data as an
        array of shape(*, 3) that forms a rough plane that intersects the
        provided locations 3D point cloud.
    :param method: Type of interpolation of horizon, either 'linear' or 'nearest'
    :param triangulation: Array of indices defining the triangulation.
        Avoids computing a Delaunay triangulation if provided.

    :returns: A boolean array of shape(*, 1) where True values represent points
        in the locations array that lie below the triangulated horizon.
    """
    drapped_locations = topo_drape_elevation(
        locations, horizon, method=method, triangulation=triangulation
    )
    below_horizon = locations[:, -1] < drapped_locations[:, -1]

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
