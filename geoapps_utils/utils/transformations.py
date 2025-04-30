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
    Rotate points counterclockwise around the x then z axes, about a center point.

    :param xyz: shape(*, 2) or shape(*, 3) Input coordinates.
    :param center: len(2) or len(3) Coordinates for the center of rotation.
    :param theta: Angle of rotation in counterclockwise degree about the z-axis
        as viewed from above.
    :param phi: Angle of rotation in couterclockwise degrees around x-axis
        as viewed from the east.
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


def ccw_east_to_cw_north(azimuth: float) -> float:
    """
    Convert counterclockwise azimuth from east to clockwise from north

    :param azimuth: Azimuth angle (in xy plane) measured in counterclockwise radians
        from east.

    :returns: Azimuth angle (in xy plane) measured in clockwise degrees from north.
    """
    return (((5 * np.pi) / 2) - azimuth) % (2 * np.pi)


def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    """
    Convert cartesian to spherical coordinates.

    :param points: Array of shape (n, 3) representing x, y, z coordinates of a point
        in 3D space.

    :returns: Array of shape (n, 2) representing the azimuth and inclination angles
        in spherical coordinates. The azimuth angle is measured in radians clockwise
        from north in the range of 0 to 2pi as viewed from above, and inclination
        angle is measured in positive radians below the horizon and negative above.
    """

    magnitude = np.linalg.norm(points, axis=1)
    inclination = -1 * np.arcsin(points[:, 2] / magnitude)
    azimuth = np.sign(points[:, 1]) * (
        np.arccos(points[:, 0] / np.linalg.norm(points[:, :2], axis=1))
    )
    azimuth = ccw_east_to_cw_north(azimuth)
    return np.column_stack((magnitude, azimuth, inclination))


def spherical_normal_to_direction_and_dip(angles: np.ndarray) -> np.ndarray:
    """
    Convert normals in spherical coordinates to dip and direction of the tangent plane.

    Confines the solution to the eastern hemisphere and applies a dip
    correction for normals originally oriented in the west.

    :param angles: Array of shape (n, 2) representing the azimuth and inclination angles
        in spherical coordinates. The azimuth angle is measured in radians clockwise
        from north in the range of 0 to 2pi as viewed from above, and inclination
        angle is measured in positive radians above the horizon and negative below.

    :returns: Array of shape (n, 2) representing azimuth from 0 to pi radians
        clockwise from north as viewed from above and dip from -pi to pi in positive
        radians below the horizon and negative above.

    """
    azimuth = angles[:, 0]
    inclination = angles[:, 1]
    # inclination *= -1
    sign = np.sign(inclination)
    sign[sign == 0] = 1
    # inclination = sign * ((np.pi / 2) - np.abs(inclination))
    inclination = sign * (np.abs(inclination) - (np.pi / 2))
    #
    # greater_than_pi = azimuth > np.pi
    # inclination[greater_than_pi] = -1 * (inclination[greater_than_pi])
    # azimuth[greater_than_pi] = azimuth[greater_than_pi] - np.pi

    return np.column_stack((azimuth, inclination))


def cartesian_normal_to_direction_and_dip(normals: np.ndarray) -> np.ndarray:
    """
    Convert 3D normal vectors to dip and direction within the eastern hemisphere.

    :param normals: Array of shape (n, 3) representing the x, y, z components of a
        normal vector in 3D space.

    :returns: Array of shape (n, 2) representing azimuth from 0 to pi radians
        clockwise from north as viewed from above and dip from -pi to pi in positive
        radians below the horizon and negative above.
    """

    spherical_normals = cartesian_to_spherical(normals)
    direction_and_dip = spherical_normal_to_direction_and_dip(spherical_normals[:, 1:])

    return direction_and_dip
