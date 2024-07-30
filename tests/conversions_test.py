#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from geoapps_utils.utils.conversions import hex_to_rgb, string_to_numeric


def test_hex_to_rgb():
    assert hex_to_rgb("#000000") == [0, 0, 0]
    assert hex_to_rgb("#ffffff") == [255, 255, 255]
    assert hex_to_rgb("#ff0000") == [255, 0, 0]


def test_string_to_numeric():
    assert string_to_numeric("test") == "test"
    assert string_to_numeric("2.1") == 2.1
    assert string_to_numeric("34") == 34
    assert string_to_numeric("1e-2") == 0.01
    assert string_to_numeric("1.05e2") == 105
