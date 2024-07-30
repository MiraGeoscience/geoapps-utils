#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy.spatial import Delaunay, cKDTree


class DetectionParameters(BaseModel):
    """
    Detection parameters expected by the ui.json file format.

    :param azimuth: Azimuth of the path.
    :param azimuth_tol: Tolerance for the azimuth of the path.
    :param damping: Damping factor between [0, 1] for the path roughness.
    :param min_edges: Minimum number of points in a curve.
    :param max_distance: Maximum distance between points in a curve.
    """

    azimuth: float | None = None
    azimuth_tol: float | None = None
    damping: float = 0
    min_edges: int = 1
    max_distance: float | None = None


def find_curves(
    vertices: np.ndarray,
    parts: np.ndarray,
    detection_params: DetectionParameters = DetectionParameters(),
) -> list[list[list[float]]]:
    """
    Find curves in a set of points.

    :param vertices: Vertices for points.
    :param parts: Identifier for points belong to common parts.
    :param detection_params: Detection parameters expected for the parts connection.

    :return: List of curves.
    """
    tri = Delaunay(vertices, qhull_options="QJ")
    if tri.simplices is None:
        return []

    simplices: np.ndarray = tri.simplices

    edges = np.vstack(
        (
            simplices[:, :2],
            simplices[:, 1:],
            simplices[:, ::2],
        )
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    distances = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    distance_sort = np.argsort(distances)
    edges, distances = edges[distance_sort, :], distances[distance_sort]

    if detection_params.max_distance is not None:
        edges = edges[distances <= detection_params.max_distance, :]

    # Reject edges with same vertices id
    edge_parts = parts[edges]
    edges = edges[edge_parts[:, 0] != edge_parts[:, 1]]

    if (
        detection_params.azimuth is not None
        and detection_params.azimuth_tol is not None
    ):
        ind = filter_segments_orientation(
            vertices, edges, detection_params.azimuth, detection_params.azimuth_tol
        )
        edges = edges[ind]

    # Walk edges until no more edges can be added
    mask = np.ones(vertices.shape[0], dtype=bool)
    out_curves = []

    for ind in range(edges.shape[0]):
        if not np.any(mask[edges[ind]]):
            continue

        mask[edges[ind]] = False
        path = [edges[ind]]
        path, mask = walk_edges(
            path, edges[ind], edges, vertices, detection_params.damping, mask=mask
        )
        path, mask = walk_edges(
            path, edges[ind][::-1], edges, vertices, detection_params.damping, mask=mask
        )
        if len(path) < detection_params.min_edges:
            continue

        out_curves.append(path)

    return out_curves


def filter_segments_orientation(
    vertices: np.ndarray, edges: np.ndarray, azimuth: float, azimuth_tol: float
):
    """
    Filter segments orientation.

    :param vertices: Vertices for points.
    :param edges: Edges for points.
    :param azimuth: Filter angle (degree) on segments orientation, clockwise from North.
    :param azimuth_tol: Tolerance (degree) on the azimuth.

    :return: Array of boolean.
    """
    vectors = vertices[edges[:, 1], :] - vertices[edges[:, 0], :]
    test_vector = np.array([np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth))])

    angles = np.arccos(np.dot(vectors, test_vector) / np.linalg.norm(vectors, axis=1))

    return np.logical_or(
        np.abs(angles) < np.deg2rad(azimuth_tol),
        np.abs(angles - np.pi) < np.deg2rad(azimuth_tol),
    )


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


def walk_edges(  # pylint: disable=too-many-arguments
    path: list,
    incoming: list,
    edges: np.ndarray,
    vertices: np.ndarray,
    damping: float = 0.0,
    mask: np.ndarray | None = None,
) -> tuple[list, np.ndarray]:
    """
    Find all edges connected to a point.

    :param path: Current list of edges forming a path.
    :param incoming: Incoming edge.
    :param edges: All edges.
    :param vertices: Direction of the edges.
    :param damping: Damping factor between [0, 1] for the path roughness.
    :param mask: Mask for nodes that have already been visited.

    :return: Edges connected to point.
    """
    if mask is None:
        mask = np.ones(edges.max() + 1, dtype=bool)
        mask[np.hstack(path).flatten()] = False

    if damping < 0 or damping > 1:
        raise ValueError("Damping must be between 0 and 1.")

    neighbours = np.where(
        np.any(edges == incoming[1], axis=1) & np.any(mask[edges], axis=1)
    )[0]

    if len(neighbours) == 0:
        return path, mask

    # Outgoing candidate nodes
    candidates = edges[neighbours][edges[neighbours] != incoming[1]]

    vectors = vertices[candidates, :] - vertices[incoming[1], :]
    in_vec = np.diff(vertices[incoming, :], axis=0).flatten()
    dot = np.dot(in_vec, vectors.T)

    if not np.any(dot > 0):
        return path, mask

    # Remove backward vectors
    vectors = vectors[dot > 0, :]
    candidates = candidates[dot > 0]
    dot = dot[dot > 0]

    # Compute the angle between the incoming vector and the outgoing vectors
    vec_lengths = np.linalg.norm(vectors, axis=1)
    angle = np.arccos(dot / (np.linalg.norm(in_vec) * vec_lengths) - 1e-10)

    # Minimize the torque
    sub_ind = np.argmin(angle ** (1 - damping) * vec_lengths)
    outgoing = [incoming[1], candidates[sub_ind]]
    mask[candidates[sub_ind]] = False
    path.append(outgoing)

    # Continue walking
    path, mask = walk_edges(path, outgoing, edges, vertices, damping, mask=mask)

    return path, mask


def weighted_average(  # pylint: disable=too-many-arguments, too-many-locals
    xyz_in: np.ndarray,
    xyz_out: np.ndarray,
    values: list,
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
    n = np.min([xyz_in.shape[0], n])
    assert isinstance(values, list), "Input 'values' must be a list of numpy.ndarrays"

    assert all(
        vals.shape[0] == xyz_in.shape[0] for vals in values
    ), "Input 'values' must have the same shape as input 'locations'"

    avg_values = []
    for value in values:
        sub = ~np.isnan(value)
        tree = cKDTree(xyz_in[sub, :])
        rad, ind = tree.query(xyz_out, n)
        ind = np.c_[ind]
        rad = np.c_[rad]
        rad[rad > max_distance] = np.nan

        values_interp = np.zeros(xyz_out.shape[0])
        weight = np.zeros(xyz_out.shape[0])

        for i in range(n):
            v = value[sub][ind[:, i]] / (rad[:, i] + threshold)
            values_interp = np.nansum([values_interp, v], axis=0)
            w = 1.0 / (rad[:, i] + threshold)
            weight = np.nansum([weight, w], axis=0)

        values_interp[weight > 0] = values_interp[weight > 0] / weight[weight > 0]
        values_interp[weight == 0] = np.nan
        avg_values += [values_interp]

    if return_indices:
        return avg_values, ind

    return avg_values
