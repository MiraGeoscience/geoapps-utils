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
import scipy.sparse as ssp


def z_rotation_matrix(angle: np.ndarray) -> ssp.csr_matrix:
    """
    Sparse matrix for heterogeneous vector rotation about the z axis.

    To be used in a matrix-vector product with an array of shape (n, 3)
    where n is the number of 3-component vectors.

    :param angle: Array of angles in radians for counterclockwise rotation
        about the z-axis.
    """
    n = len(angle)
    rza = np.c_[np.cos(angle), np.cos(angle), np.ones(n)].T
    rza = rza.flatten(order="F")
    rzb = np.c_[np.sin(angle), np.zeros(n), np.zeros(n)].T
    rzb = rzb.flatten(order="F")
    rzc = np.c_[-np.sin(angle), np.zeros(n), np.zeros(n)].T
    rzc = rzc.flatten(order="F")
    rot_z = ssp.diags([rzb[:-1], rza, rzc[:-1]], [-1, 0, 1])

    return rot_z


def x_rotation_matrix(angle: np.ndarray) -> ssp.csr_matrix:
    """
    Sparse matrix for heterogeneous vector rotation about the x axis.

    To be used in a matrix-vector product with an array of shape (n, 3)
    where n is the number of 3-component vectors.

    :param angle: Array of angles in radians for counterclockwise rotation
        about the x-axis.
    """
    n = len(angle)
    rxa = np.c_[np.ones(n), np.cos(angle), np.cos(angle)].T
    rxa = rxa.flatten(order="F")
    rxb = np.c_[np.zeros(n), np.sin(angle), np.zeros(n)].T
    rxb = rxb.flatten(order="F")
    rxc = np.c_[np.zeros(n), -np.sin(angle), np.zeros(n)].T
    rxc = rxc.flatten(order="F")
    rot_x = ssp.diags([rxb[:-1], rxa, rxc[:-1]], [-1, 0, 1])

    return rot_x


def y_rotation_matrix(angle: np.ndarray) -> ssp.csr_matrix:
    """
    Sparse matrix for heterogeneous vector rotation about the y axis.

    To be used in a matrix-vector product with an array of shape (n, 3)
    where n is the number of 3-component vectors.

    :param angle: Array of angles in radians for counterclockwise rotation
        about the y-axis.
    """
    n = len(angle)
    rxa = np.c_[np.cos(angle), np.ones(n), np.cos(angle)].T
    rxa = rxa.flatten(order="F")
    rxb = np.c_[-np.sin(angle), np.zeros(n), np.zeros(n)].T
    rxb = rxb.flatten(order="F")
    rxc = np.c_[np.sin(angle), np.zeros(n), np.zeros(n)].T
    rxc = rxc.flatten(order="F")
    rot_y = ssp.diags([rxb[:-2], rxa, rxc[:-2]], [-2, 0, 2])

    return rot_y


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


def ccw_east_to_cw_north(azimuth: np.ndarray) -> np.ndarray:
    """Convert counterclockwise azimuth from east to clockwise from north."""
    return (((5 * np.pi) / 2) - azimuth) % (2 * np.pi)


def inclination_to_dip(inclination: np.ndarray) -> np.ndarray:
    """Convert inclination from positive z-axis to dip from horizon."""
    return inclination - (np.pi / 2)


def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    """
    Convert cartesian to spherical coordinates.

    :param points: Array of shape (n, 3) representing x, y, z coordinates of a point
        in 3D space.

    :returns: Array of shape (n, 3) representing the magnitude, azimuth and inclination
        in spherical coordinates. The azimuth angle is measured in radians
        counterclockwise from east in the range of 0 to 2pi as viewed from above, and
        inclination angle is measured in radians from the positive z-axis.
    """

    magnitude = np.linalg.norm(points, axis=1)
    inclination = np.arccos(points[:, 2] / magnitude)
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    return np.column_stack((magnitude, azimuth, inclination))


def spherical_normal_to_direction_and_dip(angles: np.ndarray) -> np.ndarray:
    """
    Convert normals in spherical coordinates to dip and direction of the tangent plane.

    :param angles: Array of shape (n, 2) representing the azimuth and inclination angles
        of a normal vector in spherical coordinates. The azimuth angle is measured in
        radians counterclockwise from east in the range of 0 to 2pi as viewed from above,
        and inclination angle is measured in radians from the positive z-axis.

    :returns: Array of shape (n, 2) representing direction from 0 to 2pi radians
        clockwise from north as viewed from above and dip from -pi to pi in positive
        radians below the horizon and negative above.
    """

    rot_z = z_rotation_matrix(angles[:, 0])
    rot_y = y_rotation_matrix(angles[:, 1])
    tangents = np.tile([1, 0, 0], len(angles))
    tangents = np.reshape(rot_z * rot_y * tangents, (-1, 3))
    angles = cartesian_to_spherical(tangents)

    return np.column_stack(
        (ccw_east_to_cw_north(angles[:, 1]), inclination_to_dip(angles[:, 2]))
    )


def cartesian_normal_to_direction_and_dip(normals: np.ndarray) -> np.ndarray:
    """
    Convert 3D normal vectors to dip and direction.

    :param normals: Array of shape (n, 3) representing the x, y, z components of a
        normal vector in 3D space.

    :returns: Array of shape (n, 2) representing azimuth from 0 to 2pi radians
        clockwise from north as viewed from above and dip from -pi to pi in positive
        radians below the horizon and negative above.
    """

    spherical_normals = cartesian_to_spherical(normals)
    direction_and_dip = spherical_normal_to_direction_and_dip(spherical_normals[:, 1:])

    return direction_and_dip
