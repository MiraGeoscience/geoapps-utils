#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations


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
