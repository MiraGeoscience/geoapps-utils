# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2023-2025 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np


def rotate_xyz(xyz: np.ndarray, center: list, theta: float, phi: float = 0.0):
    """
    Perform a counterclockwise rotation of scatter points around the x-axis,
        then z-axis, about a center point.

    :param xyz: shape(*, 2) or shape(*, 3) Input coordinates.
    :param center: len(2) or len(3) Coordinates for the center of rotation.
    :param theta: Angle of rotation around z-axis in degrees.
    :param phi: Angle of rotation around x-axis in degrees.
    """
    return2d = False
    locs = xyz.copy()

    # If the input is 2-dimensional, add zeros in the z column.
    if len(center) == 2:
        center.append(0)
    if locs.shape[1] == 2:
        locs = np.concatenate((locs, np.zeros((locs.shape[0], 1))), axis=1)
        return2d = True

    locs = np.subtract(locs, center)
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)

    # Construct rotation matrix
    mat_x = np.r_[
        np.c_[1, 0, 0],
        np.c_[0, np.cos(phi), -np.sin(phi)],
        np.c_[0, np.sin(phi), np.cos(phi)],
    ]
    mat_z = np.r_[
        np.c_[np.cos(theta), -np.sin(theta), 0],
        np.c_[np.sin(theta), np.cos(theta), 0],
        np.c_[0, 0, 1],
    ]
    mat = mat_z.dot(mat_x)

    xyz_rot = mat.dot(locs.T).T
    xyz_out = xyz_rot + center

    if return2d:
        # Return 2-dimensional data if the input xyz was 2-dimensional.
        return xyz_out[:, :2]
    return xyz_out
