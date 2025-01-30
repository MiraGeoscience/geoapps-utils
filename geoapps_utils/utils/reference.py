# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from typing import Any
from uuid import UUID

from geoh5py.shared import Entity
from geoh5py.workspace import Workspace


def get_name_from_uid(workspace: Workspace, uid: Any) -> str:
    """
    Get a name from a uid.

    :param workspace: The workspace to search the UUID in.
    :param uid: The UUID to extract a name from.

    :return: the name
    """
    if not isinstance(uid, UUID):
        raise TypeError(f"Expected UUID, got {type(uid)}")

    entity = workspace.get_entity(uid)[0]

    if not isinstance(entity, Entity):
        raise ValueError(f"No Entity found in workspace for {uid}")

    return entity.name
