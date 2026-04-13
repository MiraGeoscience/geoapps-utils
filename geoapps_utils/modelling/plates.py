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
from typing import Self

import numpy as np
from geoh5py import Workspace
from geoh5py.objects import Octree, Surface
from geoh5py.objects.maxwell_plate import MaxwellPlate, PlateGeometry, PlatePosition
from geoh5py.shared.utils import fetch_active_workspace
from pydantic import BaseModel, ConfigDict, Field

from geoapps_utils.utils.transformations import (
    rotate_points,
    rotate_xyz,
    x_rotation_matrix,
    z_rotation_matrix,
)


class PlateModel(BaseModel):
    """
    Parameters describing the position and orientation of a dipping plate.

    :param strike_length: Length of the plate in the strike direction.
    :param dip_length: Length of the plate in the dip direction.
    :param width: Width of the plate.
    :param easting: Easting of the plate center.
    :param northing: Northing of the plate center.
    :param elevation: Elevation of the plate center.
    :param direction: Dip direction of the plate in degrees from North.
    :param dip: Dip angle of the plate in degrees below the horizontal.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    strike_length: float
    dip_length: float
    width: float
    easting: float
    northing: float
    elevation: float
    direction: float = Field(default=0.0, alias="dip_direction")
    dip: float = 0.0

    @classmethod
    def from_maxwell_plate_geometry(cls, geometry: PlateGeometry) -> Self:
        """Construct a PlateModel from geoh5py MaxwellPlate geometry."""

        if geometry.rotation != 0.0:
            warnings.warn(
                "Plunging plate models are not yet implemented. "
                "Ignoring the maxwell plate geometry rotation.",
                category=UserWarning,
                stacklevel=2,
            )

        return cls(
            strike_length=geometry.length,
            dip_length=geometry.width,
            width=geometry.thickness,
            easting=geometry.position.x,
            northing=geometry.position.y,
            elevation=geometry.position.z,
            direction=geometry.dip_direction,
            dip=geometry.dip,
        )

    def to_maxwell_plate_geometry(self) -> PlateGeometry:
        """Convert the PlateModel to geoh5py PlateGeometry object."""
        return PlateGeometry(
            position=PlatePosition(
                x=self.easting,
                y=self.northing,
                z=self.elevation,
            ),
            dip=self.dip,
            dip_direction=self.direction,
            length=self.strike_length,
            width=self.dip_length,
            thickness=self.width,
        )

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

    @classmethod
    def from_maxwell_plate(cls, plate: MaxwellPlate) -> Self:
        """Construct a Plate from geoh5py MaxwellPlate object."""
        if plate.geometry is None:
            raise ValueError("Maxwell plate must have its geometry set.")
        return cls(PlateModel.from_maxwell_plate_geometry(plate.geometry))

    def to_maxwell_plate(
        self, workspace: Workspace, **plate_kwargs
    ) -> MaxwellPlate:
        """
        Save the Plate as a MaxwellPlate entity in the provided workspace.

        :param workspace: Workspace to save the MaxwellPlate in.
        :param plate_kwargs: Arguments passed on to the MaxwellPlate instantiation.
        """
        with fetch_active_workspace(workspace) as ws:
            plate = MaxwellPlate.create(
                ws, geometry=self.params.to_maxwell_plate_geometry(), **plate_kwargs
            )
        return plate

    def mask(self, mesh: Octree) -> np.ndarray:
        """
        Create a mask for the centroids of the input mesh.

        :param mesh: Input mesh whose centroids will be used to define the mask.
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

        u_1 = self.params.origin[0] - (self.params.strike_length / 2.0)
        u_2 = self.params.origin[0] + (self.params.strike_length / 2.0)
        v_1 = self.params.origin[1] - (self.params.dip_length / 2.0)
        v_2 = self.params.origin[1] + (self.params.dip_length / 2.0)
        w_1 = self.params.origin[2] - (self.params.width / 2.0)
        w_2 = self.params.origin[2] + (self.params.width / 2.0)

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


def inside_plate(
    points: np.ndarray,
    plate: PlateModel,
) -> np.ndarray:
    """
    Create a plate model at a set of points from background, anomaly and size.

    :param points: Array of shape (n, 3) representing the x, y, z coordinates of the
        model space (often the cell centers of a mesh).
    :param plate: Dipping plate parameters.
    """

    xmin = plate.origin[0] - plate.strike_length / 2
    xmax = plate.origin[0] + plate.strike_length / 2
    ymin = plate.origin[1] - plate.dip_length / 2
    ymax = plate.origin[1] + plate.dip_length / 2
    zmin = plate.origin[2] - plate.width / 2
    zmax = plate.origin[2] + plate.width / 2

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
    Create a plate model at a set of points from background, anomaly, size and attitude.

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
