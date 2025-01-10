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

from copy import copy
from pathlib import Path
from typing import Any, ClassVar

from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self


class BaseData(BaseModel):
    """
    Core parameters expected by the ui.json file format.

    :param conda_environment: Environment used to run run_command.
    :param geoh5: Current workspace path.
    :param monitoring_directory: Path to monitoring directory, where .geoh5 files
        are automatically processed by GA.
    :param run_command: Command to run the application through GA.
    :param title: Application title.
    :param workspace_geoh5: Current workspace, where results will be exported.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: ClassVar[str] = "base"
    default_ui_json: ClassVar[Path | None] = None
    title: ClassVar[str] = "Base Data"
    run_command: ClassVar[str] = "geoapps_utils.driver.driver"

    conda_environment: str | None = None
    geoh5: Workspace
    monitoring_directory: str | Path | None = None
    workspace_geoh5: Path | None = None
    _input_file: InputFile | None = None

    @staticmethod
    def collect_input_from_dict(
        base_model: type[BaseModel], data: dict[str, Any]
    ) -> dict[str, dict | Any]:
        """
        Recursively replace BaseModel objects with dictionary of 'data' values.

        :param base_model: BaseModel object holding data and possibly other nested
            BaseModel objects.
        :param data: Dictionary of parameters and values without nesting structure.
        """
        update = {}
        for field, info in base_model.model_fields.items():
            if isinstance(info.annotation, type) and issubclass(
                info.annotation, BaseModel
            ):
                update[field] = BaseData.collect_input_from_dict(
                    info.annotation,
                    data,  # type: ignore
                )
            else:
                if field in data:
                    update[field] = data.get(field, info.default)

        return update

    @classmethod
    def build(cls, input_data: InputFile | dict) -> Self:
        """
        Build a dataclass from a dictionary or InputFile.

        :param input_data: Dictionary of parameters and values.

        :return: Dataclass of application parameters.
        """

        data = input_data

        if isinstance(input_data, InputFile) and input_data.data is not None:
            data = input_data.data.copy()

        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary or InputFile.")

        kwargs = BaseData.collect_input_from_dict(cls, data)  # type: ignore
        out = cls(**kwargs)
        if isinstance(input_data, InputFile):
            out._input_file = input_data

        return out

    def _recursive_flatten(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively flatten nested dictionary.

        To be used on output of BaseModel.model_dump.

        :param data: Dictionary of parameters and values.
        """
        out_dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                out_dict.update(self._recursive_flatten(value))
            else:
                out_dict.update({key: value})

        return out_dict

    def flatten(self) -> dict:
        """
        Flatten the parameters to a dictionary.

        :return: Dictionary of parameters.
        """
        out = self._recursive_flatten(self.model_dump())
        out.pop("input_file", None)

        return out

    @property
    def input_file(self) -> InputFile:
        """Create an InputFile with data matching current parameter state."""

        if self._input_file is None:
            ifile = self._create_input_file_from_attributes()
        else:
            ifile = copy(self._input_file)
            ifile.validate = False

        return ifile

    def _create_input_file_from_attributes(self) -> InputFile:
        """
        Create an InputFile with data matching current parameter state.
        """
        # ensure default uijson (PAth )exists or raise an error
        if self.default_ui_json is None or not self.default_ui_json.exists():
            raise FileNotFoundError(
                f"Default uijson file '{self.default_ui_json}' not a valid path."
            )

        ifile = InputFile.read_ui_json(self.default_ui_json, validate=False)

        attributes = self.flatten()

        ifile.data = {
            key: attributes.get(key, value) for key, value in ifile.data.items()
        }

        return ifile

    def write_ui_json(self, path: Path) -> None:
        """
        Write the ui.json file for the application.

        :param path: Path to write the ui.json file.
        """
        self.input_file.write_ui_json(path.name, str(path.parent))
