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

import numpy as np
from geoh5py import Workspace
from geoh5py.groups.property_group import GroupTypeEnum
from geoh5py.objects import Points

from geoapps_utils.utils.transformations import (
    cartesian_normal_to_direction_and_dip,
    cartesian_to_spherical,
    rotate_xyz,
    spherical_normal_to_direction_and_dip,
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


def test_cartesian_to_spherical_first_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-60, phi=-30)
    angles = np.rad2deg(cartesian_to_spherical(pole)[:, 1:])
    assert np.allclose(angles, [[60, 60]])


def test_cartesian_to_spherical_second_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-150, phi=-30)
    angles = np.rad2deg(cartesian_to_spherical(pole)[:, 1:])
    assert np.allclose(angles, [[150, 60]])


def test_cartesian_to_spherical_third_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-240, phi=-30)
    angles = np.rad2deg(cartesian_to_spherical(pole)[:, 1:])
    assert np.allclose(angles, [[240, 60]])


def test_cartesian_to_spherical_fourth_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-330, phi=-30)
    angles = np.rad2deg(cartesian_to_spherical(pole)[:, 1:])
    assert np.allclose(angles, [[330, 60]])


def test_cartesian_to_spherical_first_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-60, phi=30)
    angles = np.rad2deg(cartesian_to_spherical(pole)[:, 1:])
    assert np.allclose(angles, [[60, -60]])


def test_cartesian_to_spherical_second_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-150, phi=30)
    angles = np.rad2deg(cartesian_to_spherical(pole)[:, 1:])
    assert np.allclose(angles, [[150, -60]])


def test_cartesian_to_spherical_third_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-240, phi=30)
    angles = np.rad2deg(cartesian_to_spherical(pole)[:, 1:])
    assert np.allclose(angles, [[240, -60]])


def test_cartesian_to_spherical_fourth_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-330, phi=30)
    angles = np.rad2deg(cartesian_to_spherical(pole)[:, 1:])
    assert np.allclose(angles, [[330, -60]])


def test_spherical_to_direction_and_dip_first_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-60, phi=-30)
    spherical_coords = cartesian_to_spherical(pole)[:, 1:]
    angles = np.rad2deg(spherical_normal_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[60, 30]])


def test_spherical_to_direction_and_dip_second_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-150, phi=-30)
    spherical_coords = cartesian_to_spherical(pole)[:, 1:]
    angles = np.rad2deg(spherical_normal_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[150, 30]])


def test_spherical_to_direction_and_dip_third_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-240, phi=-30)
    spherical_coords = cartesian_to_spherical(pole)[:, 1:]
    angles = np.rad2deg(spherical_normal_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[60, -30]])


def test_spherical_to_direction_and_dip_fourth_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-330, phi=-30)
    spherical_coords = cartesian_to_spherical(pole)[:, 1:]
    angles = np.rad2deg(spherical_normal_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[150, -30]])


def test_spherical_to_direction_and_dip_first_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-60, phi=30)
    spherical_coords = cartesian_to_spherical(pole)[:, 1:]
    angles = np.rad2deg(spherical_normal_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[60, -30]])


def test_spherical_to_direction_and_dip_second_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-150, phi=30)
    spherical_coords = cartesian_to_spherical(pole)[:, 1:]
    angles = np.rad2deg(spherical_normal_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[150, -30]])


def test_spherical_to_direction_and_dip_third_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-240, phi=30)
    spherical_coords = cartesian_to_spherical(pole)[:, 1:]
    angles = np.rad2deg(spherical_normal_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[60, 30]])


def test_spherical_to_direction_and_dip_fourth_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-330, phi=30)
    spherical_coords = cartesian_to_spherical(pole)[:, 1:]
    angles = np.rad2deg(spherical_normal_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[150, 30]])


def test_spherical_values(tmp_path):
    theta, phi = np.meshgrid(np.arange(0, 360, 10), np.arange(-90, 90, 10))
    theta = theta.flatten()
    phi = phi.flatten()

    rad = 100.0
    x = rad * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
    y = rad * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    z = rad * np.sin(np.radians(phi))

    xyz = np.c_[x, y, z]
    rad_azim_incl = cartesian_to_spherical(xyz)
    assert np.allclose(rad_azim_incl[:, 0], rad)

    dirdip = cartesian_normal_to_direction_and_dip(xyz)

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
        prop_group = points.create_property_group(
            name="sphere_normals_vector",
            properties=unit_normals,
            property_group_type=GroupTypeEnum.VECTOR,
        )

        azimuth, inclination = points.add_data(
            {
                "azimuth": {"values": np.rad2deg(rad_azim_incl[:, 1])},
                "inclination": {"values": np.rad2deg(rad_azim_incl[:, 2])},
            }
        )
        prop_group = points.create_property_group(
            name="sphere_normals_dip_dir",
            properties=[azimuth, inclination],
            property_group_type=GroupTypeEnum.DIPDIR,
        )

        direction, dip = points.add_data(
            {
                "direction": {"values": np.rad2deg(dirdip[:, 0])},
                "dip": {"values": np.rad2deg(dirdip[:, 1])},
            }
        )
        prop_group = points.create_property_group(
            name="direction_dip",
            properties=[direction, dip],
            property_group_type=GroupTypeEnum.DIPDIR,
        )
