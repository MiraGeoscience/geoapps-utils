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
from pydantic import BaseModel, ConfigDict
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree

from geoapps_utils.utils.transformations import (
    rotate_points,
    x_rotation_matrix,
    z_rotation_matrix,
)


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


class PlateModel(BaseModel):
    """
    Options for creating a dipping plate model.

    :param strike_length: Length of the plate in the strike direction.
    :param dip_length: Length of the plate in the dip direction.
    :param width: Width of the plate.
    :param origin: Origin point of the plate in the form [x, y, z].
    :param direction: Dip direction of the plate in degrees from North.
    :param dip: Dip angle of the plate in degrees below the horizontal.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strike_length: float
    dip_length: float
    width: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: float = 0.0
    dip: float = 0.0


def inside_plate(
    points: np.ndarray,
    plate: PlateModel,
) -> np.ndarray:
    """
    Create a plate model at a set of points from background, anomaly and size.

    :param points: Array of shape (n, 3) representing the x, y, z coordinates of the
        model space (often the cell centers of a mesh).

    """

    xmin = plate.origin[0] - plate.strike_length / 2
    xmax = plate.origin[0] + plate.strike_length / 2
    ymin = plate.origin[1] - plate.dip_length / 2
    ymax = plate.origin[1] + plate.dip_length / 2
    zmin = plate.origin[2] - plate.width / 2
    zmax = plate.origin[2] + plate.width / 2

    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )

    return mask


def make_plate(
    points: np.ndarray,
    plate: PlateModel,
    background: float | np.ndarray = 0.0,
    anomaly: float = 1.0,
):
    """
    Create a plate model at a set of points from background, anomaly, size and attitude.

    :param points: Array of shape (n, 3) representing the x, y, z coordinates of the
        model space (often the cell centers of a mesh).
    :param plate: PlateModel object containing the parameters for the plate model.
    :param background: Background value for the model. Can be an existing model, or a value
        to be filled everywhere outside the plate.
    :param background: Background value for the model. Can be an existing model, or a value.
    :param anomaly: Value to fill inside the plate.
    """

    if isinstance(background, float):
        model = np.ones(len(points)) * background
    else:
        model = background.copy()

    rotations = [
        z_rotation_matrix(np.deg2rad(plate.direction)),
        x_rotation_matrix(np.deg2rad(plate.dip)),
    ]
    rotated_centers = rotate_points(points, origin=plate.origin, rotations=rotations)

    model[inside_plate(rotated_centers, plate)] = anomaly

    return model
