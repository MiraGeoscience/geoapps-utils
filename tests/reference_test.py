#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import pytest
from geoh5py import Workspace
from geoh5py.objects import Points

from geoapps_utils.utils.reference import get_name_from_uid


def test_get_name_from_uid(tmp_path):
    h5path = tmp_path / "test.geoh5"

    with Workspace(h5path) as workspace:
        points = Points.create(
            workspace, name="test_points", vertices=[[0, 0, 0], [1, 1, 1]]
        )
        Points.create(workspace, name="test_points", vertices=[[0, 0, 0], [1, 1, 1]])
        res = get_name_from_uid(workspace, points.uid)
        assert res == points.name

        res = get_name_from_uid(workspace, points)
        assert res == points.name

        assert "bidon" == get_name_from_uid(workspace, "bidon")

        with pytest.raises(AttributeError, match="No object with name"):
            _ = get_name_from_uid(workspace, 123)
