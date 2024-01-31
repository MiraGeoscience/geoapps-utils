#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np
from geoh5py.objects import DrapeModel, Octree
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree


def find_curves(  # pylint: disable=too-many-locals
    vertices: np.ndarray,
    ids: np.ndarray,
    min_edges: int,
    max_distance: float,
    damping: float,
) -> list[list[list[float]]]:
    """
    Find curves in a set of points.

    :param vertices: Vertices for points.
    :param ids: Ids for points.
    :param min_edges: Minimum number of points in a curve.
    :param max_distance: Maximum distance between points in a curve.
    :param damping: Damping factor between [0, 1] for the path roughness.

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
    edges = edges[distances <= max_distance, :]

    # Reject edges with same vertices id
    edge_ids = ids[edges]
    edges = edges[edge_ids[:, 0] != edge_ids[:, 1]]

    # Walk edges until no more edges can be added
    mask = np.ones(vertices.shape[0], dtype=bool)
    out_curves = []

    for ind in range(edges.shape[0]):
        if not np.any(mask[edges[ind]]):
            continue

        mask[edges[ind]] = False
        path = [edges[ind]]
        path, mask = walk_edges(path, edges[ind], edges, vertices, damping, mask=mask)
        path, mask = walk_edges(
            path, edges[ind][::-1], edges, vertices, damping, mask=mask
        )
        if len(path) < min_edges:
            continue

        out_curves.append(path)

    return out_curves


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


def cell_size_z(drape_model: DrapeModel) -> np.ndarray:
    """Compute z cell sizes of drape model."""
    hz = []
    for prism in drape_model.prisms:
        top_z, top_layer, n_layers = prism[2:]
        bottoms = drape_model.layers[
            range(int(top_layer), int(top_layer + n_layers)), 2
        ]
        z = np.hstack([top_z, bottoms])
        hz.append(z[:-1] - z[1:])
    return np.hstack(hz)


def active_from_xyz(
    mesh: DrapeModel | Octree,
    topo: np.ndarray,
    grid_reference="center",
    method="linear",
):
    """Returns an active cell index array below a surface

    :param mesh: Mesh object
    :param topo: Array of xyz locations
    :param grid_reference: Cell reference. Must be "center", "top", or "bottom"
    :param method: Interpolation method. Must be "linear", or "nearest"
    """

    mesh_dim = 2 if isinstance(mesh, DrapeModel) else 3
    locations = mesh.centroids.copy()

    if method == "linear":
        delaunay_2d = Delaunay(topo[:, :-1])
        z_interpolate = LinearNDInterpolator(delaunay_2d, topo[:, -1])
    elif method == "nearest":
        z_interpolate = NearestNDInterpolator(topo[:, :-1], topo[:, -1])
    else:
        raise ValueError("Method must be 'linear', or 'nearest'")

    if mesh_dim == 2:
        z_offset = cell_size_z(mesh) / 2.0
    else:
        z_offset = mesh.octree_cells["NCells"] * np.abs(mesh.w_cell_size) / 2

    # Shift cell center location to top or bottom of cell
    if grid_reference == "top":
        locations[:, -1] += z_offset
    elif grid_reference == "bottom":
        locations[:, -1] -= z_offset
    elif grid_reference == "center":
        pass
    else:
        raise ValueError("'grid_reference' must be one of 'center', 'top', or 'bottom'")

    z_locations = z_interpolate(locations[:, :2])

    # Apply nearest neighbour if in extrapolation
    ind_nan = np.isnan(z_locations)
    if any(ind_nan):
        tree = cKDTree(topo)
        _, ind = tree.query(locations[ind_nan, :])
        z_locations[ind_nan] = topo[ind, -1]

    # fill_nan(locations, z_locations, filler=topo[:, -1])

    # Return the active cell array
    return locations[:, -1] < z_locations


def truncate_locs_depths(locs: np.ndarray, depth_core: float) -> np.ndarray:
    """
    Sets locations below core to core bottom.

    :param locs: Location points.
    :param depth_core: Depth of core mesh below locs.

    :return locs: locs with depths truncated.
    """
    zmax = locs[:, -1].max()  # top of locs
    below_core_ind = (zmax - locs[:, -1]) > depth_core
    core_bottom_elev = zmax - depth_core
    locs[
        below_core_ind, -1
    ] = core_bottom_elev  # sets locations below core to core bottom
    return locs


def minimum_depth_core(
    locs: np.ndarray, depth_core: float, core_z_cell_size: int
) -> float:
    """
    Get minimum depth core.

    :param locs: Location points.
    :param depth_core: Depth of core mesh below locs.
    :param core_z_cell_size: Cell size in z direction.

    :return depth_core: Minimum depth core.
    """
    zrange = locs[:, -1].max() - locs[:, -1].min()  # locs z range
    if depth_core >= zrange:
        return depth_core - zrange + core_z_cell_size
    else:
        return depth_core
