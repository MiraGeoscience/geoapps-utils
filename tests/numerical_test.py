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
import pytest
from numpy import random

from geoapps_utils.utils.numerical import (
    fibonacci_series,
    fit_circle,
    running_mean,
    traveling_salesman,
    weighted_average,
)


def test_running_mean():
    vec = np.random.randn(100)
    mean_forw = running_mean(vec, method="forward")
    mean_back = running_mean(vec, method="backward")
    mean_cent = running_mean(vec, method="centered")

    mean_test = (vec[1:] + vec[:-1]) / 2

    assert np.linalg.norm(mean_back[:-1] - mean_test) < 1e-12, (
        "Backward averaging does not match expected values."
    )
    assert np.linalg.norm(mean_forw[1:] - mean_test) < 1e-12, (
        "Forward averaging does not match expected values."
    )
    assert (
        np.linalg.norm((mean_test[1:] + mean_test[:-1]) / 2 - mean_cent[1:-1]) < 1e-12
    ), "Centered averaging does not match expected values."


def test_traveling_salesman():
    x = np.linspace(0, np.pi, 50)
    y = np.sin(x)
    locs = np.zeros((len(x), 3))
    locs[:, 0] = x
    locs[:, 1] = y
    shuffled_locs = locs.copy()
    random.shuffle(shuffled_locs)
    inds = traveling_salesman(shuffled_locs)

    assert np.all(shuffled_locs[inds] == locs) or np.all(
        np.flip(shuffled_locs[inds], axis=0) == locs
    )


def test_weighted_average_same_point():
    # in loc == out loc -> in val == out val
    xyz_in = np.array([[0, 0, 0]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([99])]
    out = weighted_average(xyz_in, xyz_out, values)
    assert out[0] == 99


def test_weighted_average_same_distance():
    # two point same distance away -> arithmetic mean
    xyz_in = np.array([[1, 0, 0], [0, 1, 0]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([99, 100])]
    out = weighted_average(xyz_in, xyz_out, values)
    assert isinstance(out, list)
    assert (out[0] - 99.5) < 1e-10


def test_weighted_average_two_far_points():
    # two points different distance away but close to infinity -> arithmetic mean
    xyz_in = np.array([[1e30, 0, 0], [1e30 + 1, 0, 0]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([99, 100])]
    out = weighted_average(xyz_in, xyz_out, values)
    assert isinstance(out, list)
    assert (out[0] - 99.5) < 1e-10


def test_weighted_average_one_far_point():
    # one point close to infinity, one not -> out val is near-field value
    xyz_in = np.array([[1, 0, 0], [1e30, 0, 0]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([99, 100])]
    out = weighted_average(xyz_in, xyz_out, values)
    assert isinstance(out, list)
    assert (out[0] - 99.0) < 1e-10


def test_weighted_average_diff_length_locs():
    # one values vector and n out locs -> one out vector of length 20
    xyz_in = np.random.rand(10, 3)
    xyz_out = np.random.rand(20, 3)
    values = [np.random.rand(10)]
    out = weighted_average(xyz_in, xyz_out, values)
    assert len(out) == 1
    assert len(out[0]) == 20


def test_weighted_average_two_values_vecs():
    # two values vectors and n out locs -> two out vectors of length 20 each
    xyz_in = np.random.rand(10, 3)
    xyz_out = np.random.rand(20, 3)
    values = [np.random.rand(10), np.random.rand(10)]
    out = weighted_average(xyz_in, xyz_out, values)
    assert len(out) == 2
    assert len(out[0]) == 20
    assert len(out[1]) == 20


def test_weighted_average_max_distance():
    # max distance keeps out points from average
    xyz_in = np.array([[1, 0, 0], [3, 0, 0]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([1, 100])]
    out = weighted_average(xyz_in, xyz_out, values, max_distance=2)
    assert out[0] == 1


def test_weighted_average_n():
    # n caps the number of points that go into the average
    xyz_in = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([1, 2, 3])]
    out = weighted_average(xyz_in, xyz_out, values, n=3)
    assert out[0] == 2
    out = weighted_average(xyz_in, xyz_out, values, n=2)
    assert out[0] == 1.5


def test_weighted_average_return_indices():
    # return indices with n=1 returns closest in loc to the out loc
    xyz_in = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 2]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([1, 2, 3])]
    _, ind = weighted_average(xyz_in, xyz_out, values, n=1, return_indices=True)
    assert ind[0][0] == 1


def test_weighted_average_errors():
    expected_results = [
        np.array([[2, 0, 0], [0, 1, 0], [0, 0, 2]]),
        np.array([[0, 0, 0]]),
        [np.array([1])],
    ]

    for idx in range(len(expected_results)):
        temp_results = expected_results.copy()
        temp_results[idx] = "not expected type"

        with pytest.raises(TypeError, match="Inputs 'xyz_in'"):
            _ = weighted_average(*temp_results)  # type: ignore

    with pytest.raises(ValueError, match="Input 'values'"):
        _ = weighted_average(
            np.array([[2, 0, 0], [0, 1, 0], [0, 0, 2]]),
            np.array([[0, 0, 0]]),
            [np.array([1]), np.array([2])],
        )


def test_weighted_average_threshold():
    # threshold >> r -> arithmetic mean
    xyz_in = np.array([[1, 0, 0], [0, 100, 0], [0, 0, 1000]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([1, 2, 3])]
    out = weighted_average(xyz_in, xyz_out, values, threshold=1e30)
    assert out[0] == 2


def test_fit_cicle():
    tol = 1e-1
    n_samples = 200
    x0, y0 = np.random.randn(2) * 100.0
    r = np.random.rand() * 10
    theta = np.random.randn(n_samples) * 2 * np.pi

    x = x0 + r * np.cos(theta) + np.random.randn(n_samples) * tol / 2.0
    y = y0 + r * np.sin(theta) + np.random.randn(n_samples) * tol / 2.0
    r_fit, x_fit, y_fit = fit_circle(x, y)

    np.testing.assert_allclose(x0, x_fit, atol=tol)
    np.testing.assert_allclose(y0, y_fit, atol=tol)
    np.testing.assert_allclose(r, r_fit, atol=tol)


def test_fibonacci_series():
    np.testing.assert_allclose(fibonacci_series(1), [0])
    np.testing.assert_allclose(fibonacci_series(6), [0, 1, 1, 2, 3, 5])
