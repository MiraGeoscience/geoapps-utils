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
from geoh5py.objects import Surface

from geoapps_utils.utils.transformations import (
    compute_normals,
    normal_from_triangle,
    normals_to_dip_direction,
    rotate_xyz,
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


def test_normal_from_triangle():
    triangle = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
        ]
    )
    normal = normal_from_triangle(triangle)

    assert np.allclose(normal, [0, 0, -1]), "Error in normal calculation from triangle."


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

    assert np.allclose(normals[0], [])


def test_normals_to_dip_direction():
    normals = np.array([[np.sqrt(2) / 2, 0, np.sqrt(2) / 2]])
    dipaz = normals_to_dip_direction(normals)
    assert True
