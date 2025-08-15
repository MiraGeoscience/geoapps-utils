# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                          '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
from pydantic import BaseModel, ConfigDict

from geoapps_utils.utils.transformations import (
    rotate_points,
    x_rotation_matrix,
    z_rotation_matrix,
)


class PlateModel(BaseModel):
    """
    Parameters describing the position and orientation of a dipping plate.

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
    :param plate: Dipping plate parameters.
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
