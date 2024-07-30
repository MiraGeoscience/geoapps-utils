#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np
import pytest
from numpy import random

from geoapps_utils.numerical import (
    DetectionParameters,
    filter_segments_orientation,
    find_curves,
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

    assert (
        np.linalg.norm(mean_back[:-1] - mean_test) < 1e-12
    ), "Backward averaging does not match expected values."
    assert (
        np.linalg.norm(mean_forw[1:] - mean_test) < 1e-12
    ), "Forward averaging does not match expected values."
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


def test_weighted_average_threshold():
    # threshold >> r -> arithmetic mean
    xyz_in = np.array([[1, 0, 0], [0, 100, 0], [0, 0, 1000]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([1, 2, 3])]
    out = weighted_average(xyz_in, xyz_out, values, threshold=1e30)
    assert out[0] == 2


@pytest.fixture(name="curves_data")
def curves_data_fixture() -> list:
    # Create test data
    # Survey lines
    y_array = np.linspace(0, 50, 10)
    line_ids_array = np.arange(0, len(y_array))

    curve1 = 5 * np.sin(y_array) + 10  # curve
    curve2 = 0.7 * y_array + 20  # crossing lines
    curve3 = -0.4 * y_array + 50
    curve4 = np.ones_like(y_array) * 80  # zig-zag
    curve4[3] = 85
    curve5 = [None] * (len(y_array) - 1)  # short line
    curve5[0:1] = [60, 62]  # type: ignore
    curve5[-2:-1] = [2, 4]  # type: ignore

    curves = [curve1, curve2, curve3, curve4, curve5]

    data = []
    for channel_group, curve in enumerate(curves):
        for x_coord, y_coord, line_id in zip(
            curve, y_array, line_ids_array, strict=False
        ):
            if x_coord is not None:
                data.append([x_coord, y_coord, line_id, channel_group])
    return data


def test_find_curves(curves_data: list):
    # Random shuffle the input
    data = np.array(curves_data)
    np.random.shuffle(data)

    points_data = data[:, :2]
    line_ids = data[:, 2]
    channel_groups = data[:, 3]

    result_curves = []
    parameters = DetectionParameters(
        min_edges=3,
        max_distance=15,
        damping=0.75,
    )
    for channel_group in np.unique(channel_groups):
        channel_inds = channel_groups == channel_group
        path = find_curves(
            points_data[channel_inds], np.array(line_ids)[channel_inds], parameters
        )

        if len(path) == 0:
            continue

        result_curves += path

    assert len(result_curves) == 4
    assert len(result_curves[3]) == 8

    # Test with different angle to get zig-zag line
    result_curves = []
    parameters = DetectionParameters(
        min_edges=3,
        max_distance=50,
        damping=1,
    )

    for channel_group in np.unique(channel_groups):
        channel_inds = channel_groups == channel_group
        path = find_curves(
            points_data[channel_inds], np.array(line_ids)[channel_inds], parameters
        )

        result_curves += path

    assert [len(curve) for curve in result_curves] == [9, 9, 9, 9]


def test_find_curve_orientation(curves_data: list):
    # Random shuffle the input
    data = np.array(curves_data)
    points_data = data[:, :2]
    line_ids = data[:, 2]
    channel_groups = data[:, 3]

    result_curves = []

    parameters = DetectionParameters(
        min_edges=3,
        max_distance=15,
        damping=0.75,
        azimuth=5,
        azimuth_tol=10,
    )
    for channel_group in np.unique(channel_groups):
        channel_inds = channel_groups == channel_group
        path = find_curves(
            points_data[channel_inds], np.array(line_ids)[channel_inds], parameters
        )

        if len(path) == 0:
            continue

        result_curves += path

    assert len(result_curves) == 1


def test_filter_segments_orientation():
    angles = np.arange(0, 360, 180 / 8)

    points = np.c_[
        np.sin(np.deg2rad(angles)),
        np.cos(np.deg2rad(angles)),
    ]
    points = np.r_[np.c_[0, 0], points]
    segments = np.c_[np.zeros_like(angles), np.arange(1, len(angles) + 1)].astype(int)

    ind = filter_segments_orientation(points, segments, 0, 0.1)
    np.testing.assert_allclose(angles[ind], [0, 180])

    ind = filter_segments_orientation(points, segments, 0, 30)
    np.testing.assert_allclose(angles[ind], [0, 22.5, 157.5, 180, 202.5, 337.5])

    ind = filter_segments_orientation(points, segments, -45, 30)
    np.testing.assert_allclose(angles[ind], [112.5, 135.0, 157.5, 292.5, 315.0, 337.5])

    ind = filter_segments_orientation(points, segments, -45, 360)
    assert np.all(ind)

    ind = filter_segments_orientation(points, segments, 5, 1)
    assert ~np.all(ind)
