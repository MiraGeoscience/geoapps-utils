#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  All rights reserved.

from __future__ import annotations

from uuid import UUID

from geoh5py.data import FloatData, IntegerData
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace

from geoapps_utils.utils.list import sorted_alphanumeric_list


def sorted_children_dict(
    entity: UUID | Entity, workspace: Workspace | None = None
) -> dict[str, UUID] | None:
    """
    Uses natural sorting algorithm to order the keys of a dictionary containing
    children name/uid key/value pairs.

    If valid uuid entered calls get_entity.  Will return None if no entity found
    in workspace for provided entity

    :param entity: geoh5py entity containing children IntegerData, FloatData
        entities

    :return : sorted name/uid dictionary of children entities of entity.

    """

    if isinstance(entity, UUID) and workspace is not None:
        entity = workspace.get_entity(entity)[0]

    if hasattr(entity, "children"):
        children_dict = {}
        for child in entity.children:
            if not isinstance(child, (IntegerData, FloatData)):
                continue

            children_dict[child.name] = child.uid

        children_order = sorted_alphanumeric_list(list(children_dict))

        return {k: children_dict[k] for k in children_order}
    return None
