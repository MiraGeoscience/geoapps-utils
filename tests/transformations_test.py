#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np

from geoapps_utils.utils.transformations import rotate_xyz


def test_positive_rotation_xyz():
    rot_vec = rotate_xyz(np.c_[1, 0, 0], [0, 0], 45)

    assert (
        np.linalg.norm(np.cross(rot_vec, [0.7071, 0.7071, 0])) < 1e-8
    ), "Error on positive rotation about origin."


def test_negative_rotation_xyz():
    rot_vec = rotate_xyz(np.c_[1, 0, 0], [1, 1], -90)

    assert (
        np.linalg.norm(np.cross(rot_vec, [0, 1, 0])) < 1e-8
    ), "Error on negative rotation about point."


def test_2d_input():
    assert (rotate_xyz(np.c_[1, 0], [0, 0], 45)).shape[1] == 2, "Error on 2D input."
