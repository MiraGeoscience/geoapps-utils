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


def hex_to_rgb(hex_color: str) -> list[int]:
    """
    Convert hex color code to RGB

    :param hex_color: Hex color code.

    :return rgb: RGB color code.
    """
    code = hex_color.lstrip("#")
    return [int(code[i : i + 2], 16) for i in (0, 2, 4)]


def string_to_numeric(text: str) -> int | float | str:
    """
    Converts numeric string representation to int, float, or nan if possible.

    :param text: String to be converted.

    :return: Int, float, or nan if possible, otherwise returns the original string.
    """
    try:
        text_as_float = float(text)
        text_as_int = int(text_as_float)
        return text_as_int if text_as_int == text_as_float else text_as_float
    except ValueError:
        return np.nan if text == "nan" else text
