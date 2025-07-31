# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                          '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np

from geoh5py import Workspace
from geoh5py.objects import (
    LargeLoopGroundTEMReceivers,
    LargeLoopGroundTEMTransmitters,
)

from tests.testing_utils.terrain import gaussian_topo_drape
from . import channels, waveform

def generate_tdem_survey(
    geoh5: Workspace,
    vertices: np.ndarray,
    n_lines: int,
    flatten: bool = False,
):


    X = vertices[:, 0].reshape((n_lines, -1))
    Y = vertices[:, 1].reshape((n_lines, -1))
    Z = vertices[:, 2].reshape((n_lines, -1))
    center = np.mean(vertices, axis=0)
    arrays = [
        np.c_[
            X[: int(n_lines / 2), :].flatten(),
            Y[: int(n_lines / 2), :].flatten(),
            Z[: int(n_lines / 2), :].flatten(),
        ],
        np.c_[
            X[int(n_lines / 2) :, :].flatten(),
            Y[int(n_lines / 2) :, :].flatten(),
            Z[int(n_lines / 2) :, :].flatten(),
        ],
    ]
    loops = []
    loop_cells = []
    loop_id = []
    count = 0
    for ind, array in enumerate(arrays):
        loop_id += [np.ones(array.shape[0]) * (ind + 1)]
        min_loc = np.min(array, axis=0)
        max_loc = np.max(array, axis=0)
        loop = np.vstack(
            [
                np.c_[
                    np.ones(5) * min_loc[0],
                    np.linspace(min_loc[1], max_loc[1], 5),
                ],
                np.c_[
                    np.linspace(min_loc[0], max_loc[0], 5)[1:],
                    np.ones(4) * max_loc[1],
                ],
                np.c_[
                    np.ones(4) * max_loc[0],
                    np.linspace(max_loc[1], min_loc[1], 5)[1:],
                ],
                np.c_[
                    np.linspace(max_loc[0], min_loc[0], 5)[1:-1],
                    np.ones(3) * min_loc[1],
                ],
            ]
        )
        loop = (loop - np.mean(loop, axis=0)) * 1.5 + np.mean(loop, axis=0)
        loop = np.c_[
            loop, gaussian_topo_drape(loop[:, 0], loop[:, 1], flatten=flatten)
        ]
        loops += [loop + np.asarray(center)]
        loop_cells += [np.c_[np.arange(15) + count, np.arange(15) + count + 1]]
        loop_cells += [np.c_[count + 15, count]]
        count += 16

    transmitters = LargeLoopGroundTEMTransmitters.create(
        geoh5,
        vertices=np.vstack(loops),
        cells=np.vstack(loop_cells),
    )
    transmitters.tx_id_property = transmitters.parts + 1
    survey = LargeLoopGroundTEMReceivers.create(geoh5, vertices=np.vstack(vertices))
    survey.transmitters = transmitters
    survey.tx_id_property = np.hstack(loop_id)
    # survey.parts = np.repeat(np.arange(n_lines), n_electrodes)

    survey.channels = channels

    survey.waveform = waveform
    survey.timing_mark = 2.0
    survey.unit = "Milliseconds (ms)"

    return survey