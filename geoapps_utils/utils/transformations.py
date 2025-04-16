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
from geoh5py.objects import Surface
# from simpeg.utils.mat_utils import cartesian2amplitude_dip_azimuth


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


def normal_from_triangle(triangle: np.ndarray) -> np.ndarray:
    """Compute the normal vector from a triangle defined by three vertices."""
    v1 = triangle[1] - triangle[0]
    v2 = triangle[2] - triangle[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.zeros(3)
    return normal / norm


def compute_normals(surface: Surface) -> np.ndarray:
    """Compute normals for each triangle in a surface."""
    normals = []
    for cell in surface.cells:
        triangle = surface.vertices[cell, :]
        normal = normal_from_triangle(triangle)
        normals.append(normal)

    return np.vstack(normals)

def cartesian2spherical(m):
    r"""
    Converts a set of 3D vectors from Cartesian to spherical coordinates.

    Parameters
    ----------
    m : (n, 3) array_like
        An array whose columns represent the x, y and z components of
        a set of vectors.

    Returns
    -------
    (n, 3) numpy.ndarray
        An array whose columns represent the *a*, *t* and *p* components
        of a set of vectors in spherical coordinates.

    Notes
    -----

    In Cartesian space, the components of each vector are defined as

    .. math::

        \mathbf{v} = (v_x, v_y, v_z)

    In spherical coordinates, vectors are is defined as:

    .. math::

        \mathbf{v^\prime} = (a, t, p)

    where

        - :math:`a` is the amplitude of the vector
        - :math:`t` is the azimuthal angle defined positive from vertical
        - :math:`p` is the radial angle defined positive CCW from Easting

    """

    # nC = int(len(m)/3)

    x = m[:, 0]
    y = m[:, 1]
    z = m[:, 2]

    a = (x**2.0 + y**2.0 + z**2.0) ** 0.5

    t = np.zeros_like(x)
    t[a > 0] = np.arcsin(z[a > 0] / a[a > 0])

    p = np.zeros_like(x)
    p[a > 0] = np.arctan2(y[a > 0], x[a > 0])

    m_atp = np.r_[a, t, p]

    return m_atp


def cartesian2amplitude_dip_azimuth(m):
    """
    Convert from cartesian to amplitude, dip (positive down) and
    azimuth (clockwise for North), in degree.
    """
    m = m.reshape((-1, 3), order="F")
    atp = cartesian2spherical(m).reshape((-1, 3), order="F")
    atp[:, 1] = np.rad2deg(-1.0 * atp[:, 1])
    atp[:, 2] = (450.0 - np.rad2deg(atp[:, 2])) % 360.0

    return atp

def normals_to_dip_direction(normals: np.ndarray):
    return cartesian2amplitude_dip_azimuth(normals)

