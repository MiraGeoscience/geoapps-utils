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
from geoh5py.objects import Surface


def z_rotation_matrix(angle: np.ndarray | float) -> ssp.csr_matrix:
    """
    Sparse matrix for heterogeneous vector rotation about the z axis.

    To be used in a matrix-vector product with an array of shape (n, 3)
    where n is the number of 3-component vectors.

    :param angle: Array of angles in radians for counterclockwise rotation
        about the z-axis.
    """

    if isinstance(angle, np.ndarray):
        n = len(angle)
        rza = np.c_[np.cos(angle), np.cos(angle), np.ones(n)].T
        rza = rza.flatten(order="F")
        rzb = np.c_[np.sin(angle), np.zeros(n), np.zeros(n)].T
        rzb = rzb.flatten(order="F")
        rzc = np.c_[-np.sin(angle), np.zeros(n), np.zeros(n)].T
        rzc = rzc.flatten(order="F")
        rot_z = ssp.diags([rzb[:-1], rza, rzc[:-1]], [-1, 0, 1])
    else:
        rot_z = np.r_[
            np.c_[np.cos(angle), -np.sin(angle), 0],
            np.c_[np.sin(angle), np.cos(angle), 0],
            np.c_[0, 0, 1],
        ]

    return rot_z


def x_rotation_matrix(angle: np.ndarray | float) -> ssp.csr_matrix:
    """
    Sparse matrix for heterogeneous vector rotation about the x axis.

    To be used in a matrix-vector product with an array of shape (n, 3)
    where n is the number of 3-component vectors.

    :param angle: Array of angles in radians for counterclockwise rotation
        about the x-axis.
    """

    if isinstance(angle, np.ndarray):
        n = len(angle)
        rxa = np.c_[np.ones(n), np.cos(angle), np.cos(angle)].T
        rxa = rxa.flatten(order="F")
        rxb = np.c_[np.zeros(n), np.sin(angle), np.zeros(n)].T
        rxb = rxb.flatten(order="F")
        rxc = np.c_[np.zeros(n), -np.sin(angle), np.zeros(n)].T
        rxc = rxc.flatten(order="F")
        rot_x = ssp.diags([rxb[:-1], rxa, rxc[:-1]], [-1, 0, 1])
    else:
        rot_x = np.r_[
            np.c_[1, 0, 0],
            np.c_[0, np.cos(angle), -np.sin(angle)],
            np.c_[0, np.sin(angle), np.cos(angle)],
        ]

    return rot_x


def y_rotation_matrix(angle: np.ndarray | float) -> ssp.csr_matrix:
    """
    Sparse matrix for heterogeneous vector rotation about the y axis.

    To be used in a matrix-vector product with an array of shape (n, 3)
    where n is the number of 3-component vectors.

    :param angle: Array of angles in radians for counterclockwise rotation
        about the y-axis.
    """

    if isinstance(angle, np.ndarray):
        n = len(angle)
        rxa = np.c_[np.cos(angle), np.ones(n), np.cos(angle)].T
        rxa = rxa.flatten(order="F")
        rxb = np.c_[-np.sin(angle), np.zeros(n), np.zeros(n)].T
        rxb = rxb.flatten(order="F")
        rxc = np.c_[np.sin(angle), np.zeros(n), np.zeros(n)].T
        rxc = rxc.flatten(order="F")
        rot_y = ssp.diags([rxb[:-2], rxa, rxc[:-2]], [-2, 0, 2])
    else:
        rot_y = np.r_[
            np.c_[np.cos(angle), 0, np.sin(angle)],
            np.c_[0, 1, 0],
            np.c_[-np.sin(angle), 0, np.cos(angle)],
        ]

    return rot_y


def apply_rotation(operator: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply heterogeneous or homogeneous rotation matrix to point set.

    :param operator: Rotation matrix of shape (3, 3) or (n, 3, 3) where n is the number of
        points.
    :param points: Array of shape (n, 3) representing the x, y, z coordinates to be rotated.
    """

    if operator.shape[0] == points.shape[1]:
        return operator.dot(points.T).T

    if operator.shape[0] == np.prod(points.shape):
        vec = points.copy()
        vec = vec.flatten()
        vec = operator.dot(vec.T).T
        vec = vec.reshape(-1, 3)
        return vec

    raise ValueError(
        f"Shape mismatch between operator ({operator.shape}) and points "
        f"({points.shape}). For n points, the rotation operator should be size n "
        "if homogeneous, or 3n if heterogeneous."
    )


def rotate_points(
    points: np.ndarray,
    origin: tuple[float, float, float],
    rotations: list[ssp.csr_matrix],
) -> np.ndarray:
    """
    Rotate points through a series of rotations about the provided origin.

    :param points: Array of shape (n, 3) representing the x, y, z coordinates.
    :param origin: Origin point of the rotation in the form [x, y, z].
    :param rotations: List of rotation matrices to apply to the points.  These must
        be in the form of scipy sparse matrices (csr_matrix) produced by the
        x_rotation_matrix(), y_rotation_matrix(), and z_rotation_matrix() functions.
    """

    out = points.copy() - origin
    for rotation in rotations:
        out = apply_rotation(rotation, out)
    return out + origin


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

    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    xyz_out = rotate_points(
        locs,
        tuple(center),
        rotations=[x_rotation_matrix(phi), z_rotation_matrix(theta)],
    )

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
    Convert cartesian to spherical coordinate system.

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


def compute_normals(surface: Surface) -> np.ndarray:
    """
    Compute normals for each triangle in a surface.

    :param surface: Surface object containing vertices and cells.

    :returns: Array of shape (n, 3) representing the normals for
        each cell in the mesh.
    """

    vertices = surface.vertices
    cells = surface.cells

    v1 = vertices[cells[:, 1]] - vertices[cells[:, 0]]
    v2 = vertices[cells[:, 2]] - vertices[cells[:, 0]]
    normals = np.cross(v1, v2)
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    return normals


def azimuth_to_unit_vector(azimuth: float) -> np.ndarray:
    """
    Convert an azimuth to a unit vector.

    :param azimuth: Azimuth in degrees from north (0 to 360).
    :return: Unit vector in the direction of the azimuth.
    """
    theta = np.deg2rad(azimuth)
    mat_z = np.r_[
        np.c_[np.cos(theta), -np.sin(theta), 0.0],
        np.c_[np.sin(theta), np.cos(theta), 0.0],
        np.c_[0.0, 0.0, 1.0],
    ]
    return np.array([0.0, 1.0, 0.0]).dot(mat_z)
