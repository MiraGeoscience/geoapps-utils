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

from geoapps_utils.utils.formatters import recursive_flatten, string_name


def test_string_name():
    chars = "!@#$%^&*().,"
    value = "H!e(l@l#o.W$o%r^l&d*"
    assert string_name(value, characters=chars) == "H_e_l_l_o_W_o_r_l_d_", (
        "string_name validator failed"
    )


def test_recursive_flatten():
    data = {
        "a": 1,
        "b": {"c": 2, "d": {"e": 3}},
        "f": 4,
    }
    expected = {
        "a": 1,
        "c": 2,
        "e": 3,
        "f": 4,
    }
    assert recursive_flatten(data) == expected, "recursive_flatten failed"
