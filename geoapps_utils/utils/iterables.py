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

import re
from uuid import UUID

from geoh5py.data import FloatData, IntegerData
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace

from geoapps_utils.utils.conversions import string_to_numeric


def find_value(labels: list, keywords: list, default=None) -> list:
    """
    Find matching keywords within a list of labels.

    :param labels: List of labels or list of [key, value] that may contain the keywords.
    :param keywords: List of keywords to search for.
    :param default: Default value be returned if none of the keywords are found.

    :return matching_labels: List of labels containing any of the keywords.
    """
    value = None
    for entry in labels:
        for string in keywords:
            if isinstance(entry, list):
                name = entry[0]
            else:
                name = entry

            if isinstance(string, str) and (
                (string.lower() in name.lower()) or (name.lower() in string.lower())
            ):
                if isinstance(entry, list):
                    value = entry[1]
                else:
                    value = name

    if value is None:
        value = default
    return value


def sorted_alphanumeric_list(alphanumerics: list[str]) -> list[str]:
    """
    Sorts a list of strings containing alphanumeric characters in readable way.

    Sorting precedence is alphabetical for all string components followed by
    numeric component found in string from left to right.

    :param alphanumerics: list of alphanumeric strings.

    :return : naturally sorted list of alphanumeric strings.
    """

    def sort_precedence(text):
        numeric_regex = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
        non_numeric = re.split(numeric_regex, text)
        numeric = [string_to_numeric(k) for k in re.findall(numeric_regex, text)]
        order = non_numeric + numeric
        return order

    return sorted(alphanumerics, key=sort_precedence)


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
    :param workspace: geoh5py workspace containing entity

    :return : sorted name/uid dictionary of children entities of entity.

    """
    entity_obj = entity
    if isinstance(entity, UUID) and workspace is not None:
        obj = workspace.get_entity(entity)[0]
        if isinstance(obj, Entity):
            entity_obj = obj

    if isinstance(entity_obj, Entity):
        children_dict = {}
        for child in entity_obj.children:  # type: ignore
            if not isinstance(child, IntegerData | FloatData):
                continue

            children_dict[child.name] = child.uid

        children_order = sorted_alphanumeric_list(list(children_dict))

        return {k: children_dict[k] for k in children_order}
    return None
