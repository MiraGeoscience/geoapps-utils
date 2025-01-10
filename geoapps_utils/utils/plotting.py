# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2023-2025 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np


def symlog(values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert values to log with linear threshold near zero.

    :param values: Values to convert.
    :param threshold: Linear threshold.

    :return: Converted values.
    """
    return np.sign(values) * np.log10(1 + np.abs(values) / threshold)


def inv_symlog(values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Compute the inverse symlog mapping.

    :param values: Values to convert.
    :param threshold: Linear threshold.

    :return: Converted values.
    """
    return np.sign(values) * threshold * (-1.0 + 10.0 ** np.abs(values))


def format_axis(
    channel: str, axis: np.ndarray, log: bool, threshold: float, nticks: int = 5
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
    """
    Format plot axis ticks and labels.

    :param channel: Axis name.
    :param axis: Axis values.
    :param log: Whether to plot log.
    :param threshold: Linear threshold.
    :param nticks: Number of ticks.

    :return:
    """
    label = channel

    if log:
        axis = symlog(axis, threshold)

    values = axis[~np.isnan(axis)]
    ticks = np.linspace(values.min(), values.max(), nticks)

    if log:
        label = f"Log({channel})"
        ticklabels = inv_symlog(ticks, threshold)
    else:
        ticklabels = ticks

    return axis, label, ticks, ticklabels.tolist()
