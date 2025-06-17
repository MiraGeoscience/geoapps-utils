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

from typing import Any


def string_name(value: str, characters: str = ".") -> str:
    """
    Find and replace characters in a string with underscores '_'.

    :param value: String to be validated
    :param characters: Characters to be replaced

    :return value: Re-formatted string
    """
    for char in characters:
        value = value.replace(char, "_")
    return value


def recursive_flatten(data: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively flatten nested dictionary.

    To be used on output of BaseModel.model_dump.

    :param data: Dictionary of parameters and values.
    """
    out_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out_dict.update(recursive_flatten(value))
        else:
            out_dict.update({key: value})

    return out_dict
