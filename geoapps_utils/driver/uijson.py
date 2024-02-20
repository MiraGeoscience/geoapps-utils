#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from pathlib import Path
from typing import Annotated, Literal, Optional, Union
from uuid import UUID

from geoh5py import Workspace
from geoh5py.data import Data
from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import is_uuid
from geoh5py.ui_json import InputFile
from geoh5py.ui_json.utils import path2workspace, workspace2path
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    PlainSerializer,
    field_validator,
    model_validator,
)

from .. import __version__


def parent_validator(value, info):
    if isinstance(value, Data) and value.parent is not info.data["parent"].value:
        raise ValueError(
            f"Input '{info.data['label']}' must be linked to the selected "
            f"parent '{info.data['parent'].label}'."
        )
    return value


WorkspaceConversion = Annotated[
    Union[Workspace, str, Path],
    AfterValidator(path2workspace),
    PlainSerializer(workspace2path),
]

ParentalValue = Annotated[Data, AfterValidator(parent_validator)]

EnabledMode = Literal["enable", "disable"]


class BaseForm(BaseModel):
    """
    label: Description of the parameter.
    main: If set to true, the parameter is shown in the first tab of the UI.
    tooltip: Verbose description the parameter that appears when the mouse hovers over it.
    optional: Whether the parameter is optional. On output, check if enabled is set to true.
    enabled: Whether the parameter is enabled or not (grey and inactive in the UI).
    group: Name of the group to which the parameter belongs.
    groupOptional: If true, adds a checkbox in the top of the group box next to the name.
    dependency: The name of the parameter which this parameter is dependent upon.
    dependencyType: What happens when the dependency member is checked.
    groupDependency: The name of the object of which the group of the parameter is dependent upon.
    groupDependencyType: What happens when the group’s dependency parameter is checked.
    """

    label: str
    main: Optional[bool] = None
    tooltip: Optional[str] = None
    optional: Optional[bool] = None
    enabled: bool = True
    group: Optional[str] = None
    groupOptional: Optional[bool] = None
    dependency: Optional[str] = None
    dependencyType: Optional[EnabledMode] = None
    groupDependency: Optional[str] = None
    groupDependencyType: Optional[EnabledMode] = None


class StringForm(BaseForm):
    """
    String parameter
    """

    value: str


class FloatForm(BaseForm):
    """
    String parameter
    """

    value: float
    vmin: Optional[int] = None
    vmax: Optional[int] = None


class IntegerForm(BaseForm):
    """
    String parameter
    """

    value: int
    vmin: Optional[int] = None
    vmax: Optional[int] = None


class EntityForm(BaseForm):
    """
    Form for entities linked to a workspace geoh5.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def promote(cls, form, info) -> Workspace:
        if not isinstance(info.data, dict):
            return form

        workspace = info.data.get("geoh5", None)
        if isinstance(workspace, Workspace) and is_uuid(form["value"]):
            value = UUID(str(form["value"]))
            form["value"] = workspace.get_entity(value)[0]

        return form


class ObjectSelectionForm(EntityForm):
    """
    String parameter
    """

    value: Optional[ObjectBase] = None


class DataSelectionForm(EntityForm):
    """
    String parameter
    """

    _parent_form: Optional[ObjectSelectionForm] = None

    parent: Optional[ObjectSelectionForm] = None
    value: Optional[ParentalValue] = None

    @model_validator(mode="before")
    @classmethod
    def assign_parent(cls, form, info):
        if (
            form["parent"] is not None
            and isinstance(info.data, dict)
            and form["parent"] in info.data
        ):
            form["parent"] = info.data[form["parent"]]

        return form


class BaseUIJson(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    _name: str = "base"

    input_file: Optional[InputFile] = None
    conda_environment: Optional[str] = None
    geoh5: Optional[Workspace] = None
    monitoring_directory: Optional[Union[str, Path]] = None
    run_command: str
    title: str
    workspace_geoh5: Optional[WorkspaceConversion] = None
    version: str = __version__

    @field_validator("geoh5", mode="before")
    @classmethod
    def promote_workspace(cls, value):
        if isinstance(value, (str, Path)) and Path(value).suffix == ".geoh5":
            workspace = Workspace(value, mode="r")
            return workspace
        return value

    def model_post_init(self, _):
        if isinstance(self.geoh5, Workspace):
            self.geoh5.close()
