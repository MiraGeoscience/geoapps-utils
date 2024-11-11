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

from geoapps_utils.utils.reference import get_a_from_b, get_entities


def test_get_entity(tmp_path):
    h5path = tmp_path / "test.geoh5"

    with Workspace(h5path) as workspace:
        points = Points.create(
            workspace, name="test_points", vertices=[[0, 0, 0], [1, 1, 1]]
        )
        Points.create(workspace, name="test_points", vertices=[[0, 0, 0], [1, 1, 1]])
        print(workspace.list_entities_name)
        res = get_entities(workspace, points.uid)
        assert res[0] == points

        with pytest.raises(ValueError, match=r"Multiple \(2\) entities found"):
            _ = get_entities(workspace, "test_points")

        with pytest.raises(ValueError, match="No entity found"):
            _ = get_entities(workspace, "bidon")

        with pytest.raises(TypeError, match="Entity None is not a valid Entity"):
            _ = get_entities(workspace, None)  # type: ignore


def test_get_a_from_b(tmp_path):
    h5path = tmp_path / "test.geoh5"

    with Workspace(h5path) as workspace:
        points = Points.create(
            workspace, name="test_points", vertices=[[0, 0, 0], [1, 1, 1]]
        )
        res = get_a_from_b([points], Points, "name")
        assert res[0] == points.name

        with pytest.raises(TypeError, match="Entity None is not a valid"):
            _ = get_a_from_b(None, Points, "vertices")  # type: ignore

        with pytest.raises(AttributeError, match="Attribute 'bidon' not found in"):
            _ = get_a_from_b(points, Points, "bidon")
