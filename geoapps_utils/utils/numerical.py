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
from scipy.spatial import cKDTree


def running_mean(
    values: np.ndarray, width: int = 1, method: str = "centered"
) -> np.ndarray:
    """
    Compute a running mean of an array over a defined width.

    :param values: Input values to compute the running mean over
    :param width: Number of neighboring values to be used
    :param method: Choice between 'forward', 'backward' and ['centered'] averaging.

    :return mean_values: Averaged array values of shape(values, )
    """
    # Averaging vector (1/N)
    weights = np.r_[np.zeros(width + 1), np.ones_like(values)]
    sum_weights = np.cumsum(weights)

    mean = np.zeros_like(values)

    # Forward averaging
    if method in ["centered", "forward"]:
        padded = np.r_[np.zeros(width + 1), values]
        cumsum = np.cumsum(padded)
        mean += (cumsum[(width + 1) :] - cumsum[: (-width - 1)]) / (
            sum_weights[(width + 1) :] - sum_weights[: (-width - 1)]
        )

    # Backward averaging
    if method in ["centered", "backward"]:
        padded = np.r_[np.zeros(width + 1), values[::-1]]
        cumsum = np.cumsum(padded)
        mean += (
            (cumsum[(width + 1) :] - cumsum[: (-width - 1)])
            / (sum_weights[(width + 1) :] - sum_weights[: (-width - 1)])
        )[::-1]

    if method == "centered":
        mean /= 2.0

    return mean


def traveling_salesman(locs: np.ndarray) -> np.ndarray:
    """
    Finds the order of a roughly linear point set.
    Uses the point furthest from the mean location as the starting point.
    :param: locs: Cartesian coordinates of points lying either roughly within a plane or a line.
    :param: return_index: Return the indices of the end points in the original array.
    """
    mean = locs[:, :2].mean(axis=0)
    current = np.argmax(np.linalg.norm(locs[:, :2] - mean, axis=1))
    order = [current]
    mask = np.ones(locs.shape[0], dtype=bool)
    mask[current] = False

    for _ in range(locs.shape[0] - 1):
        remaining = np.where(mask)[0]
        ind = np.argmin(np.linalg.norm(locs[current, :2] - locs[remaining, :2], axis=1))
        current = remaining[ind]
        order.append(current)
        mask[current] = False

    return np.asarray(order)


def weighted_average(
    xyz_in: np.ndarray,
    xyz_out: np.ndarray,
    values: list[np.ndarray],
    *,
    max_distance: float = np.inf,
    n: int = 8,
    return_indices: bool = False,
    threshold: float = 1e-1,
) -> list | tuple[list, np.ndarray]:
    """
    Perform a inverse distance weighted averaging on a list of values.

    :param xyz_in: shape(*, 3) Input coordinate locations.
    :param xyz_out: shape(*, 3) Output coordinate locations.
    :param values: Values to be averaged from the input to output locations.
    :param max_distance: Maximum averaging distance beyond which values do not
        contribute to the average.
    :param n: Number of nearest neighbours used in the weighted average.
    :param return_indices: If True, return the indices of the nearest neighbours
        from the input locations.
    :param threshold: Small value added to the radial distance to avoid zero division.
        The value can also be used to smooth the interpolation.

    :return avg_values: List of values averaged to the output coordinates
    """
    if (
        not isinstance(xyz_in, np.ndarray)
        or not isinstance(xyz_out, np.ndarray)
        or not isinstance(values, list)
        or not all(isinstance(val, np.ndarray) for val in values)
    ):
        raise TypeError(
            "Inputs 'xyz_in' and 'xyz_out' must be numpy.ndarrays "
            "and 'values' must be a list of numpy.ndarrays."
            f"Got {type(xyz_in)} and {type(xyz_out)} for 'xyz_in' and 'xyz_out' "
            f"respectively, and {type(values)} for 'values'."
        )

    if not all(vals.shape[0] == xyz_in.shape[0] for vals in values):
        raise ValueError(
            "Input 'values' must have the same number of rows as input 'xyz_in'. "
            f"Got {xyz_in.shape[0]} for 'xyz_in' and "
            f"{[val.shape[0] for val in values]} for 'values'."
        )

    n = np.min([xyz_in.shape[0], n])

    avg_values = []
    for value in values:
        sub = ~np.isnan(value)
        rad, ind = cKDTree(xyz_in[sub]).query(xyz_out, n)

        if n == 1:
            ind = ind[:, np.newaxis]
            rad = rad[:, np.newaxis]

        rad = np.where(rad > max_distance, np.nan, rad) + threshold

        values_interp = np.nansum(value[sub][ind] / rad, axis=1)
        weight = np.nansum(1.0 / rad, axis=1)

        avg_values.append(values_interp / weight)

    if return_indices:
        return avg_values, ind

    return avg_values


def fibonacci_series(n: int) -> np.ndarray:
    """
    Generate Fibonacci series up to n.

    :param n: Maximum value of the series.

    :return: Fibonacci series.
    """
    a, b = 0, 1
    series = []
    while a < n:
        series.append(a)
        a, b = b, a + b
    return np.array(series)


def fit_circle(x_val: np.ndarray, y_val) -> tuple[float, float, float]:
    """
    Compute the least-square circle fit to a set of points.

    Forms a linear system of equations to solve for the circle parameters, where
    the equation of a circle is given by:

    (x - x0)^2 + (y - y0)^2 = r^2

    or

    x^2 + y^2 - 2*x0*x - 2*y0*y + x0^2 + y0^2 = r^2

    The linear system is then given by:

    [2*x, 2*y, 1][x0, y0, c] = x^2 + y^2

    where c = x0^2 + y0^2 - r^2

    :param x_val: x-coordinates of the points
    :param y_val: y-coordinates of the points

    :return (radius, x0, y0):
        Tuple of values representing the radius and center of the circle.
    """
    # Build linear system
    lin_eqs = np.c_[x_val * 2, y_val * 2, np.ones_like(x_val)]

    # Right-hand side
    rhs = (x_val**2.0 + y_val**2.0).reshape((-1, 1))

    # Find the least-square solution
    coef, _, _, _ = np.linalg.lstsq(lin_eqs, rhs, rcond=None)

    # Compute radius
    radius = (coef[0] ** 2.0 + coef[1] ** 2.0 + coef[2]) ** 0.5

    return radius, coef[0], coef[1]
