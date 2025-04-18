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

from geoapps_utils.utils.transformations import (
    cartesian_to_spherical,
    rotate_xyz,
    spherical_to_direction_and_dip,
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
    angles = np.rad2deg(cartesian_to_spherical(pole))
    assert np.allclose(angles, [[60, 60]])


def test_cartesian_to_spherical_second_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-150, phi=-30)
    angles = np.rad2deg(cartesian_to_spherical(pole))
    assert np.allclose(angles, [[150, 60]])


def test_cartesian_to_spherical_third_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-240, phi=-30)
    angles = np.rad2deg(cartesian_to_spherical(pole))
    assert np.allclose(angles, [[240, 60]])


def test_cartesian_to_spherical_fourth_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-330, phi=-30)
    angles = np.rad2deg(cartesian_to_spherical(pole))
    assert np.allclose(angles, [[330, 60]])


def test_cartesian_to_spherical_first_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-60, phi=30)
    angles = np.rad2deg(cartesian_to_spherical(pole))
    assert np.allclose(angles, [[60, -60]])


def test_cartesian_to_spherical_second_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-150, phi=30)
    angles = np.rad2deg(cartesian_to_spherical(pole))
    assert np.allclose(angles, [[150, -60]])


def test_cartesian_to_spherical_third_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-240, phi=30)
    angles = np.rad2deg(cartesian_to_spherical(pole))
    assert np.allclose(angles, [[240, -60]])


def test_cartesian_to_spherical_fourth_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-330, phi=30)
    angles = np.rad2deg(cartesian_to_spherical(pole))
    assert np.allclose(angles, [[330, -60]])


def test_spherical_to_direction_and_dip_first_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-60, phi=-30)
    spherical_coords = cartesian_to_spherical(pole)
    angles = np.rad2deg(spherical_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[60, 30]])


def test_spherical_to_direction_and_dip_second_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-150, phi=-30)
    spherical_coords = cartesian_to_spherical(pole)
    angles = np.rad2deg(spherical_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[150, 30]])


def test_spherical_to_direction_and_dip_third_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-240, phi=-30)
    spherical_coords = cartesian_to_spherical(pole)
    angles = np.rad2deg(spherical_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[60, -30]])


def test_spherical_to_direction_and_dip_fourth_quadrant_upwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, 1], center=[0, 0, 0], theta=-330, phi=-30)
    spherical_coords = cartesian_to_spherical(pole)
    angles = np.rad2deg(spherical_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[150, -30]])


def test_spherical_to_direction_and_dip_first_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-60, phi=30)
    spherical_coords = cartesian_to_spherical(pole)
    angles = np.rad2deg(spherical_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[60, -30]])


def test_spherical_to_direction_and_dip_second_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-150, phi=30)
    spherical_coords = cartesian_to_spherical(pole)
    angles = np.rad2deg(spherical_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[150, -30]])


def test_spherical_to_direction_and_dip_third_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-240, phi=30)
    spherical_coords = cartesian_to_spherical(pole)
    angles = np.rad2deg(spherical_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[60, 30]])


def test_spherical_to_direction_and_dip_fourth_quadrant_downwards():
    pole = rotate_xyz(xyz=np.c_[0, 0, -1], center=[0, 0, 0], theta=-330, phi=30)
    spherical_coords = cartesian_to_spherical(pole)
    angles = np.rad2deg(spherical_to_direction_and_dip(spherical_coords))
    assert np.allclose(angles, [[150, 30]])
