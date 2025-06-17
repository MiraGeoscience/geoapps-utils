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

import inspect
import logging
import sys
from abc import ABC, abstractmethod
from copy import copy
from importlib import import_module
from json import load
from pathlib import Path
from typing import Any, ClassVar, GenericAlias  # type: ignore

from geoh5py import Workspace
from geoh5py.groups import UIJsonGroup
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile, monitored_directory_copy
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from geoapps_utils.driver.params import BaseParams
from geoapps_utils.utils.formatters import recursive_flatten
from geoapps_utils.utils.importing import GeoAppsError


logger = logging.getLogger()


class Driver(ABC):
    """
    # todo: Get rid of BaseParams to have a more robust DriverClass

    Base driver class.

    :param params: Application parameters.
    """

    _params_class: type[Options | BaseParams]
    _validations: dict | None = None

    def __init__(self, params: Options | BaseParams):
        self._out_group: UIJsonGroup | None = None
        self.params = params

    @property
    def params(self):
        """Application parameters."""
        return self._params

    @params.setter
    def params(self, val: Options):
        if not isinstance(val, Options | BaseParams):
            raise TypeError(
                "Parameters must be of type BaseParams or Options,"
                f" get {type(val)} instead."
            )
        self._params = val

    @property
    def workspace(self):
        """Application workspace."""
        return self._params.geoh5

    @property
    def out_group(self) -> UIJsonGroup | None:
        if self._out_group is None:
            if self.params.out_group is not None:
                self._out_group = self.params.out_group
        return self._out_group

    @property
    def params_class(self):
        """Default parameter class."""
        return self._params_class

    @abstractmethod
    def run(self):
        """Run the application."""
        raise NotImplementedError

    @classmethod
    def start(cls, filepath: str | Path, driver_class=None, **kwargs):
        """
        Run application specified by 'filepath' ui.json file.

        :param filepath: Path to valid ui.json file for the application driver.
        :param driver_class: Application driver class.
        :param kwargs: Additional keyword arguments for InputFile read_ui_json.
        """

        if driver_class is None:
            driver_class = cls

        print("Loading input file . . .")
        filepath = Path(filepath).resolve()
        ifile = InputFile.read_ui_json(filepath, validations=cls._validations, **kwargs)
        with ifile.geoh5.open(mode="r+"):
            try:
                params = driver_class._params_class.build(ifile)
                print("Initializing application . . .")
                driver = driver_class(params)
                print("Running application . . .")
                driver.run()
                print(f"Results saved to {params.geoh5.h5file}")

                return driver
            except GeoAppsError as error:
                logger.warning("\n\nApplicationError: %s\n\n", error)
                sys.exit(1)

    def add_ui_json(self, entity: ObjectBase):
        """
        Add ui.json file to entity.

        :param entity: Object to add ui.json file to.
        """
        if (
            self.params.input_file is None
            or self.params.input_file.path is None
            or self.params.input_file.name is None
        ):
            raise ValueError("Input file and it's name and path must be set.")

        entity.add_file(
            str(Path(self.params.input_file.path) / self.params.input_file.name)
        )

    def update_monitoring_directory(self, entity: ObjectBase):
        """
        If monitoring directory is active, copy entity to monitoring directory.

        :param entity: Object being added to monitoring directory.
        """
        self.add_ui_json(entity)
        if (
            self.params.monitoring_directory is not None
            and Path(self.params.monitoring_directory).is_dir()
        ):
            monitored_directory_copy(
                str(Path(self.params.monitoring_directory).resolve()), entity
            )


class Options(BaseModel):
    """
    Core parameters expected by the ui.json file format.

    :param conda_environment: Environment used to run run_command.
    :param geoh5: Current workspace path.
    :param monitoring_directory: Path to monitoring directory, where .geoh5 files
        are automatically processed by GA.
    :param run_command: Command to run the application through GA.
    :param title: Application title.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: ClassVar[str] = "base"
    default_ui_json: ClassVar[Path | None] = None

    title: str = "Base Data"
    run_command: str = "geoapps_utils.base"
    conda_environment: str | None = None
    geoh5: Workspace
    monitoring_directory: str | Path | None = None
    out_group: UIJsonGroup | None = None
    _input_file: InputFile | None = None

    @staticmethod
    def collect_input_from_dict(
        model: type[BaseModel], data: dict[str, Any]
    ) -> dict[str, dict | Any]:
        """
        Recursively replace BaseModel objects with nested dictionary of 'data' values.

        :param base_model: BaseModel object to structure data for.
        :param data: Flat dictionary of parameters and values without nesting structure.
        """
        update = data.copy()
        nested_fields: list[str] = []
        for field, info in model.model_fields.items():
            # Already a BaseModel, no need to nest
            if isinstance(update.get(field, None), BaseModel):
                continue

            if (
                isinstance(info.annotation, type)
                and not isinstance(info.annotation, GenericAlias)
                and issubclass(info.annotation, BaseModel)
            ):
                # Nest and deal with aliases
                update = Options.collect_input_from_dict(info.annotation, update)
                nested = info.annotation.model_construct(**update).model_dump(
                    exclude_unset=True
                )

                if any(nested):
                    update[field] = nested
                    nested_fields += nested

        for field in nested_fields:
            if field in update:
                del update[field]

        return update

    @classmethod
    def build(cls, input_data: InputFile | dict | None = None, **kwargs) -> Self:
        """
        Build a dataclass from a dictionary or InputFile.

        :param input_data: Dictionary of parameters and values.

        :return: Dataclass of application parameters.
        """
        data = input_data or {}
        if isinstance(input_data, InputFile) and input_data.data is not None:
            data = input_data.data.copy()

        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary or InputFile.")

        data.update(kwargs)
        options = Options.collect_input_from_dict(cls, data)  # type: ignore
        out = cls(**options)

        if isinstance(input_data, InputFile):
            out._input_file = input_data

        return out

    def _recursive_flatten(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively flatten nested dictionary.

        To be used on output of BaseModel.model_dump.

        :param data: Dictionary of parameters and values.
        """
        logger.warning(
            "Deprecated method: Use geoapps_utils.utils.formatters._recursive_flatten"
        )
        return recursive_flatten(data)

    def flatten(self) -> dict:
        """
        Flatten the parameters to a dictionary.

        :return: Dictionary of parameters.
        """
        out = recursive_flatten(self.model_dump())
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
            ifile = InputFile(
                ui_json=recursive_flatten(self.model_dump()), validate=False
            )
        else:
            ifile = InputFile.read_ui_json(self.default_ui_json, validate=False)

        if ifile.data is None:
            raise ValueError(
                f"Input file {self.default_ui_json} does not contain any data."
            )

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

    def serialize(self):
        """Return a demoted uijson dictionary representation the params data."""

        dump = self.model_dump(exclude_unset=True)
        dump["geoh5"] = str(dump["geoh5"].h5file.resolve())
        ifile = self.input_file
        ifile.update_ui_values(recursive_flatten(dump))
        assert ifile.ui_json is not None
        options = ifile.stringify(ifile.demote(ifile.ui_json))

        return options


def fetch_driver_class(json_dict: str | Path | dict) -> type[Driver]:
    """
    Fetch the driver class from the ui.json 'run_command'.

    :param filepath: Path to a ui.json file with a 'run_command' key.
    """
    # TODO Remove after deprecation of geoapps_utils.driver
    from geoapps_utils.driver.driver import (  # pylint: disable=import-outside-toplevel, cyclic-import
        BaseDriver,
    )

    if isinstance(json_dict, (str, Path)):
        with open(json_dict, encoding="utf-8") as jsonfile:
            uijson = load(jsonfile)
    else:
        uijson = json_dict

    if not isinstance(uijson, dict) or "run_command" not in uijson:
        raise ValueError(
            f"Invalid ui.json file: {json_dict}. It must contain a 'run_command' key."
        )

    module = import_module(uijson["run_command"])

    cls = None
    for _, cls in inspect.getmembers(module):
        try:
            if (
                issubclass(cls, Driver | BaseDriver)
                and cls.__module__ == module.__name__
            ):
                break
        except TypeError:
            continue

    else:
        raise ValueError(
            f"No valid driver class found in module {uijson['run_command']}"
        )

    return cls


if __name__ == "__main__":
    file = sys.argv[1]
    driver_cls = fetch_driver_class(file)
    driver_cls.start(file)
