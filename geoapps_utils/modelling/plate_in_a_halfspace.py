# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                          '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from pathlib import Path

import numpy as np
from geoh5py import Workspace
from geoh5py.objects import Points, Surface
from scipy.spatial import Delaunay

from geoapps_utils.modelling.plates import PlateModel, make_plate
from tests.testing_utils.terrain import gaussian_topo_drape


plate_model_default = PlateModel(
    strike_length=40.0,
    dip_length=40.0,
    width=40.0,
    origin=(0.0, 0.0, 10.0),
)


def setup_inversion_workspace(
    work_dir,
    plate_model: PlateModel = plate_model_default,
    background=None,
    anomaly=None,
    cell_size=(5.0, 5.0, 5.0),
    center=(0.0, 0.0, 0.0),
    n_electrodes=20,
    n_lines=5,
    refinement=(4, 6),
    x_limits=(-100.0, 100.0),
    y_limits=(-100.0, 100.0),
    padding_distance=100,
    drape_height=5.0,
    inversion_type="other",
    flatten=False,
    geoh5=None,
):
    """
    Creates a workspace populated with objects to simulate/invert a simple model.

    rot_xyz: Rotation angles in degrees about the x, y, and z axes.
    """
    filepath = Path(work_dir) / "inversion_test.ui.geoh5"
    if geoh5 is None:
        if filepath.is_file():
            filepath.unlink()
        geoh5 = Workspace.create(filepath)
    # Topography
    xx, yy = np.meshgrid(np.linspace(-200.0, 200.0, 50), np.linspace(-200.0, 200.0, 50))

    zz = gaussian_topo_drape(xx, yy, flatten=flatten)
    topo = np.c_[
        utils.mkvc(xx) + center[0],
        utils.mkvc(yy) + center[1],
        utils.mkvc(zz) + center[2],
    ]
    triang = Delaunay(topo[:, :2])
    topography = Surface.create(
        geoh5,
        vertices=topo,
        cells=triang.simplices,  # pylint: disable=E1101
        name="topography",
    )

    # Observation points
    n_electrodes = (
        4
        if ("polarization" in inversion_type or "current" in inversion_type)
        & (n_electrodes < 4)
        else n_electrodes
    )
    xr = np.linspace(x_limits[0], x_limits[1], int(n_electrodes))
    yr = np.linspace(y_limits[0], y_limits[1], int(n_lines))
    X, Y = np.meshgrid(xr, yr)
    Z = gaussian_topo_drape(X, Y, flatten=flatten) + drape_height

    vertices = np.c_[
        utils.mkvc(X.T) + center[0],
        utils.mkvc(Y.T) + center[1],
        utils.mkvc(Z.T) + center[2],
    ]

    if "polarization" in inversion_type or "current" in inversion_type:
        survey = generate_dc_survey(geoh5, X, Y, Z)

    elif "magnetotellurics" in inversion_type:
        survey = MTReceivers.create(
            geoh5,
            vertices=vertices,
            name="survey",
            components=[
                "Zxx (real)",
                "Zxx (imag)",
                "Zxy (real)",
                "Zxy (imag)",
                "Zyx (real)",
                "Zyx (imag)",
                "Zyy (real)",
                "Zyy (imag)",
            ],
            channels=[10.0, 100.0, 1000.0],
        )

    elif "tipper" in inversion_type:
        survey = TipperReceivers.create(
            geoh5,
            vertices=vertices,
            name="survey",
            components=[
                "Txz (real)",
                "Txz (imag)",
                "Tyz (real)",
                "Tyz (imag)",
            ],
        )
        survey.base_stations = TipperBaseStations.create(
            geoh5, vertices=np.c_[vertices[0, :]].T
        )
        survey.channels = [10.0, 100.0, 1000.0]
        dist = np.linalg.norm(
            survey.vertices[survey.cells[:, 0], :]
            - survey.vertices[survey.cells[:, 1], :],
            axis=1,
        )
        # survey.cells = survey.cells[dist < 100.0, :]
        survey.remove_cells(np.where(dist > 200)[0])

    elif inversion_type in ["fdem", "fem", "fdem 1d"]:
        survey = generate_fdem_survey(geoh5, vertices)

    elif "tdem" in inversion_type:
        survey = generate_tdem_survey(
            geoh5,
            vertices,
            n_lines,
            flatten=flatten,
            airborne="airborne" in inversion_type,
        )

    else:
        survey = Points.create(
            geoh5,
            vertices=vertices,
            name="survey",
        )

    # Create a mesh
    if "2d" in inversion_type:
        lines = survey.get_entity("line_ids")[0].values
        entity, mesh, _ = get_drape_model(  # pylint: disable=unbalanced-tuple-unpacking
            geoh5,
            "Models",
            survey.vertices[np.unique(survey.cells[lines == 101, :]), :],
            [cell_size[0], cell_size[2]],
            100.0,
            [padding_distance] * 2 + [padding_distance] * 2,
            1.1,
            parent=None,
            return_colocated_mesh=True,
            return_sorting=True,
        )
        active = active_from_xyz(entity, topography.vertices, grid_reference="top")

    else:
        padDist = np.ones((3, 2)) * padding_distance
        mesh = mesh_builder_xyz(
            vertices - np.r_[cell_size] / 2.0,
            cell_size,
            depth_core=100.0,
            padding_distance=padDist,
            mesh_type="TREE",
            tree_diagonal_balance=False,
        )
        mesh = OctreeDriver.refine_tree_from_surface(
            mesh,
            topography,
            levels=refinement,
            finalize=False,
        )

        if inversion_type in ["fdem", "airborne tdem"]:
            mesh = OctreeDriver.refine_tree_from_points(
                mesh,
                vertices,
                levels=[2],
                finalize=False,
            )

        mesh.finalize()
        entity = treemesh_2_octree(geoh5, mesh, name="mesh")
        active = active_from_xyz(entity, topography.vertices, grid_reference="top")

    # Create the Model
    cell_centers = entity.centroids.copy()

    model = make_plate(
        cell_centers, plate_model, background=background, anomaly=anomaly
    )

    if "1d" in inversion_type:
        model = background * np.ones(mesh.nC)
        model[(mesh.cell_centers[:, 2] < 0) & (mesh.cell_centers[:, 2] > -20)] = anomaly

    model[~active] = np.nan
    model = entity.add_data({"model": {"values": model}})
    geoh5.close()
    return geoh5, entity, model, survey, topography
