# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2023-2026 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import numpy as np
import pytest
from geoh5py import Workspace
from geoh5py.groups.property_group import GroupTypeEnum
from geoh5py.objects import Points, Surface

from geoapps_utils.utils.transformations import (
    azimuth_to_unit_vector,
    cartesian_normal_to_direction_and_dip,
    cartesian_to_spherical,
    compute_normals,
    rotate_xyz,
    spherical_normal_to_direction_and_dip,
    x_rotation_matrix,
    xyz_to_polar,
    z_rotation_matrix,
)


def test_positive_rotation_xyz():
    rot_vec = rotate_xyz(np.c_[1, 0, 0], [0, 0], 45)

    assert np.linalg.norm(np.cross(rot_vec, [0.7071, 0.7071, 0])) < 1e-8, (
        "Error on positive rotation about origin."
    )


def test_negative_rotation_xyz():
    rot_vec = rotate_xyz(np.c_[1, 0, 0], [1, 1], -90)

    assert np.linalg.norm(np.cross(rot_vec, [0, 1, 0])) < 1e-8, (
        "Error on negative rotation about point."
    )


def test_2d_input():
    assert (rotate_xyz(np.c_[1, 0], [0, 0], 45)).shape[1] == 2, "Error on 2D input."


testdata = [
    (-60, -30, 1, [30, 30]),
    (-150, -30, 1, [-60, 30]),
    (-240, -30, 1, [-150, 30]),
    (-330, -30, 1, [120, 30]),
    (-60, 30, -1, [30, 150]),
    (-150, 30, -1, [-60, 150]),
    (-240, 30, -1, [-150, 150]),
    (-330, 30, -1, [120, 150]),
]


@pytest.mark.parametrize("theta,phi,polarity,expected", testdata)
def test_cartesian_to_spherical(theta, phi, polarity, expected):
    pole = rotate_xyz(xyz=np.c_[0, 0, polarity], center=[0, 0, 0], theta=theta, phi=phi)
    angles = np.rad2deg(cartesian_to_spherical(pole)[:, 1:])
    assert np.allclose(angles, [expected])


testdata = [
    (-60, -30, 1, [60, 30]),
    (-150, -30, 1, [150, 30]),
    (-240, -30, 1, [240, 30]),
    (-330, -30, 1, [330, 30]),
    (-60, 30, -1, [240, 30]),
    (-150, 30, -1, [330, 30]),
    (-240, 30, -1, [60, 30]),
    (-330, 30, -1, [150, 30]),
]


@pytest.mark.parametrize("theta,phi,polarity,expected", testdata)
def test_spherical_to_direction_and_dip_upwards(theta, phi, polarity, expected):
    pole = rotate_xyz(xyz=np.c_[0, 0, polarity], center=[0, 0, 0], theta=theta, phi=phi)
    spherical_coords = cartesian_to_spherical(pole)[:, 1:]
    angles = np.rad2deg(spherical_normal_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [expected])


def test_spherical_values(tmp_path):  # pylint: disable=too-many-locals
    theta, phi = np.meshgrid(np.arange(0, 360, 10), np.arange(-90, 90, 10))
    theta = theta.flatten()
    phi = phi.flatten()

    rad = 100.0
    x = rad * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
    y = rad * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    z = rad * np.sin(np.radians(phi))

    xyz = np.c_[x, y, z]
    rad_azim_incl = cartesian_to_spherical(xyz)
    dirdip = cartesian_normal_to_direction_and_dip(xyz)
    assert np.allclose(rad_azim_incl[:, 0], rad)
    rot_x = x_rotation_matrix(-1 * dirdip[:, 1])
    rot_z = z_rotation_matrix(-1 * dirdip[:, 0])
    tangents = np.reshape(rot_z * rot_x * np.tile([0, 1, 0], len(dirdip)), (-1, 3))
    assert np.allclose(np.linalg.norm(np.cross(xyz, tangents), axis=1), 100)

    # Create a workspace and add the data for visual inspection
    with Workspace.create(tmp_path / f"{__name__}.geoh5") as workspace:
        points = Points.create(
            workspace,
            name="points_on_sphere",
            vertices=xyz,
        )

        unit_normals = points.add_data(
            {
                "x": {"values": x / rad},
                "y": {"values": y / rad},
                "z": {"values": z / rad},
            }
        )
        _ = points.create_property_group(
            name="sphere_normals_vector",
            properties=unit_normals,
            property_group_type=GroupTypeEnum.VECTOR,
        )

        _ = points.add_data(
            {
                "azimuth": {"values": np.rad2deg(rad_azim_incl[:, 1])},
                "inclination": {"values": np.rad2deg(rad_azim_incl[:, 2])},
            }
        )

        direction, dip = points.add_data(
            {
                "direction": {"values": np.rad2deg(dirdip[:, 0])},
                "dip": {"values": np.rad2deg(dirdip[:, 1])},
            }
        )
        _ = points.create_property_group(
            name="direction_dip",
            properties=[direction, dip],
            property_group_type=GroupTypeEnum.DIPDIR,
        )


def create_surface(workspace, upside_down=False):
    k = -5.0 if upside_down else 5.0
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [5.0, 5.0, k],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ]
    )
    surface = Surface.create(workspace, vertices=vertices, cells=triangles)

    return surface


def test_compute_normals(tmp_path):
    ws = Workspace(tmp_path / "test.geoh5")
    surface = create_surface(ws, upside_down=False)
    normals = compute_normals(surface)

    h = np.sqrt(2) / 2
    validation = np.array([[0, -h, h], [h, 0, h], [0, h, h], [-h, 0, h]])
    assert np.allclose(normals, validation)


def test_azimuth_to_unit_vector():
    assert np.allclose(azimuth_to_unit_vector(0.0), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(azimuth_to_unit_vector(90.0), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(azimuth_to_unit_vector(180.0), np.array([0.0, -1.0, 0.0]))
    assert np.allclose(azimuth_to_unit_vector(270.0), np.array([-1.0, 0.0, 0.0]))
    assert np.allclose(azimuth_to_unit_vector(360.0), np.array([0.0, 1.0, 0.0]))


def test_xyz_to_polar():
    """
    Test the xyz_to_polar utility function.
    """

    rad = np.arange(0, 100, 10)
    azm = np.pi / 3.0

    # Create x, y, z coordinates
    x = rad * np.cos(azm)
    y = rad * np.sin(azm)
    z = np.random.randn(x.shape[0])  # Random z values
    locations = np.c_[x, y, z]

    # Start reference
    polar = xyz_to_polar(locations)
    np.testing.assert_almost_equal(polar[0, 0], 0.0)  # First point at zero distance
    np.testing.assert_allclose(
        polar[:, 1], np.rad2deg(azm)
    )  # All other distances positive

    # Mean reference locations
    polar = xyz_to_polar(locations - np.mean(locations, axis=0))
    np.testing.assert_almost_equal(
        polar[4::-1, 0], polar[5:, 0]
    )  # Opposite side of center
    np.testing.assert_almost_equal(polar[:5, 1] % 180, polar[5:, 1])  # Polar opposite
    np.testing.assert_allclose(z - np.mean(z), polar[:, 2])  # Preserves z

    # End reference
    polar = xyz_to_polar(locations - locations[-1, :])
    np.testing.assert_allclose(
        polar[:, 1], np.rad2deg(azm) + 180
    )  # All other distances positive
