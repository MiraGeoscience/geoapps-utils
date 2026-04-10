# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2022-2026 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import warnings

import numpy as np
from geoh5py import Workspace
from geoh5py.objects import Octree, Surface
from geoh5py.shared.utils import fetch_active_workspace
from pydantic import BaseModel, ConfigDict, Field, model_validator

from geoapps_utils.utils.transformations import (
    rotate_points,
    rotate_xyz,
    x_rotation_matrix,
    z_rotation_matrix,
)


class PlateModel(BaseModel):
    """
    Parameters describing the position and orientation of a dipping plate.

    Dip rotations are applied about the plate's origin (easting, northing,
    and elevation) so that the origin of a dipping plate marks the center
    of the plate's top (up-dip) face.  Without dip rotations, the plate is
    horizontal striking east-west with the origin at the center of the
    southern face.


    :param strike_length: Length of the plate in the strike direction.
    :param dip_length: Length of the plate in the dip direction.
    :param width: Width of the plate.
    :param easting: Easting of the center of the plate's top face.
    :param northing: Northing of the center of the plate's top face.
    :param elevation: Elevation of the center of the plate's top face.
    :param direction: Dip direction of the plate in degrees from North.
    :param dip: Dip angle of the plate in degrees below the horizontal.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    strike_length: float
    dip_length: float
    width: float
    easting: float = 0.0
    northing: float = 0.0
    elevation: float = 0.0
    direction: float = Field(default=0.0, alias="dip_direction")
    dip: float = 0.0

    @model_validator(mode="after")
    def check_origin_set(self):
        if not all(
            k in self.model_fields_set for k in ["easting", "northing", "elevation"]
        ):
            warnings.warn(
                "Not all origin parameters ('easting', 'northing', 'elevation') were set. "
                "Missing parameters default to 0 and may lead to unexpected results."
            )
        return self

    @property
    def origin(self) -> tuple[float, float, float]:
        return (self.easting, self.northing, self.elevation)


class Plate:
    """
    Plate representation for surface extraction and masking.

    :param params: Parameters describing the plate.
    """

    def __init__(self, params: PlateModel):
        self.params = params

    def mask(self, mesh: Octree) -> np.ndarray:
        """
        Return a mask for generating models with a plate anomaly.

        :param mesh: Octree mesh object defining cell centers on which
            the mask will be defined.

        :return Boolean mask that can be applied to models on the cell
            centers of the input mesh
        """
        rotations = [
            z_rotation_matrix(np.deg2rad(self.params.direction)),
            x_rotation_matrix(np.deg2rad(self.params.dip)),
        ]
        rotated_centers = rotate_points(
            mesh.centroids, origin=self.params.origin, rotations=rotations
        )
        return inside_plate(rotated_centers, self.params)

    def surface(self, workspace: Workspace, name: str = "plate") -> Surface:
        """
        Create a rectangular prism geoh5py.Surface representing the plate.

        :param workspace: Workspace object to save the surface in.
        :param name: Name of the surface.
        """

        with fetch_active_workspace(workspace) as ws:
            surface = Surface.create(
                ws,
                vertices=self.vertices,
                cells=self.triangles,
                name=name,
            )

        return surface

    @property
    def triangles(self) -> np.ndarray:
        """Triangulation of the block."""
        return np.vstack(
            [
                [0, 2, 1],
                [1, 2, 3],
                [0, 1, 4],
                [4, 1, 5],
                [1, 3, 5],
                [5, 3, 7],
                [2, 6, 3],
                [3, 6, 7],
                [0, 4, 2],
                [2, 4, 6],
                [4, 5, 6],
                [6, 5, 7],
            ]
        )

    @property
    def vertices(self) -> np.ndarray:
        """Vertices for triangulation of a rectangular prism in 3D space."""

        u_1, u_2, v_1, v_2, w_1, w_2 = bounding_box(
            origin=list(self.params.origin),
            strike_length=self.params.strike_length,
            dip_length=self.params.dip_length,
            width=self.params.width,
        )

        vertices = np.array(
            [
                [u_1, v_1, w_1],
                [u_2, v_1, w_1],
                [u_1, v_2, w_1],
                [u_2, v_2, w_1],
                [u_1, v_1, w_2],
                [u_2, v_1, w_2],
                [u_1, v_2, w_2],
                [u_2, v_2, w_2],
            ]
        )

        return self._rotate(vertices)

    def _rotate(self, vertices: np.ndarray) -> np.ndarray:
        """Rotate vertices and adjust for reference point."""
        theta = -1 * self.params.direction
        phi = -1 * self.params.dip
        rotated_vertices = rotate_xyz(vertices, list(self.params.origin), theta, phi)

        return rotated_vertices


def bounding_box(
    origin: list[float], strike_length: float, dip_length: float, width: float
) -> list[float]:
    """
    Calculate unrotated bounding box from plate geometry.

    :param origin: Southern face of an east-west striking horizontal plate.
    :param strike_length: Length of the plate in the strike (x) dimension.
    :param dip_length: Length of the plate in the (0) dip (y) dimension.
    :param width: Width of the plate (z dimension).
    """
    xmin = origin[0] - strike_length / 2
    xmax = origin[0] + strike_length / 2
    ymin = origin[1]
    ymax = origin[1] + dip_length
    zmin = origin[2] - width / 2
    zmax = origin[2] + width / 2

    return [xmin, xmax, ymin, ymax, zmin, zmax]


def inside_plate(
    points: np.ndarray,
    plate: PlateModel,
) -> np.ndarray:
    """
    Create a mask to identify input points located inside the parameterized plate.

    The plate is treated as orthogonal to the coordinate axes, and any rotation
    parameters in the PlateModel are ignored. For rotated plates, create or wrap a
    Plate from the plate parameters and use Plate.mask instead.

    The plate is treated as orthogonal to the coordinate axes, and any rotation
    parameters in the PlateModel are ignored.  For masking rotated plates, consider
    constructing a Plate object to use it's mask method.

    :param points: Array of shape (n, 3) representing the x, y, z coordinates of the
        model space (often the cell centers of a mesh).
    :param plate: Dipping plate parameters.
    """

    xmin, xmax, ymin, ymax, zmin, zmax = bounding_box(
        origin=list(plate.origin),
        strike_length=plate.strike_length,
        dip_length=plate.dip_length,
        width=plate.width,
    )

    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )

    return mask


def make_plate(
    points: np.ndarray,
    plate: PlateModel,
    background: float | np.ndarray = 0.0,
    anomaly: float = 1.0,
):
    """
    Create a plate model at a set of points from background, anomaly, size and geometry.

    :param points: Array of shape (n, 3) representing the x, y, z coordinates of the
        model space (often the cell centers of a mesh).
    :param plate: PlateModel object containing the parameters for the plate model.
    :param background: Background value for the model. Can be an existing model, or a value
        to be filled everywhere outside the plate.
    :param background: Background value for the model. Can be an existing model, or a value.
    :param anomaly: Value to fill inside the plate.
    """

    if isinstance(background, float):
        model = np.ones(len(points)) * background
    else:
        model = background.copy()

    rotations = [
        z_rotation_matrix(np.deg2rad(plate.direction)),
        x_rotation_matrix(np.deg2rad(plate.dip)),
    ]
    rotated_centers = rotate_points(points, origin=plate.origin, rotations=rotations)

    model[inside_plate(rotated_centers, plate)] = anomaly

    return model
