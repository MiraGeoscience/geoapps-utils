#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import Any
from uuid import UUID

from geoh5py.groups import PropertyGroup
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace


def get_entities(
    workspace: Workspace,
    entities: str
    | UUID
    | Entity
    | PropertyGroup
    | list[str | UUID | Entity | PropertyGroup],
) -> list[Entity]:
    """
    Get a list of entities from a workspace.

    :param workspace: Workspace containing the entities.
    :param entities: Entity or list of entities to get.

    :return: a list of entities.
    """

    if not isinstance(entities, list):
        entities = [entities]

    out_list = []
    for entity in entities:
        if isinstance(entity, str | UUID):
            temp = workspace.get_entity(entity)
            if len(temp) > 1:
                raise ValueError(f"Multiple ({len(temp)}) entities found for {entity}.")
            if len(temp) == 0 or temp[0] is None:
                raise ValueError(f"No entity found for {entity}.")
            entity = temp[0]
        if not isinstance(entity, Entity):
            raise TypeError(f"Entity {entity} is not a valid Entity.")
        out_list.append(entity)

    return out_list


def get_a_from_b(entities: list[Entity], data_type: type, attr: str) -> list[Any]:
    """
    Get a list of attributes from a list of entities.

    :param entities: List of entities.
    :param data_type: The type of data to extract.
    :param attr: The attribute to extract.

    :return: A list of attributes.
    """
    if not isinstance(entities, list):
        entities = [entities]

    out_list = []
    for entity in entities:
        if not isinstance(entity, data_type):
            raise TypeError(f"Entity {entity} is not a valid {data_type}.")
        if not hasattr(entity, attr):
            raise AttributeError(f"Attribute '{attr}' not found in {entity}.")
        out_list.append(getattr(entity, attr))

    return out_list
