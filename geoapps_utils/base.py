# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2022-2026 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from __future__ import annotations

import sys
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, GenericAlias, Self  # type: ignore

from geoh5py import Workspace
from geoh5py.groups import UIJsonGroup
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile, UIJson, monitored_directory_copy
from geoh5py.ui_json.utils import fetch_active_workspace
from pydantic import BaseModel, ConfigDict, ValidationError

from geoapps_utils import assets_path
from geoapps_utils.utils.formatters import recursive_flatten
from geoapps_utils.utils.importing import GeoAppsError
from geoapps_utils.utils.logger import get_logger


logger = get_logger(name=__name__, level_name=False, propagate=False, add_name=False)


def input_file_deprecation_warning(input_file: InputFile) -> UIJson:
    """
    Warn the user of future deprecation and get a file path to an existing file.
    """

    warnings.warn(
        "The use of InputFile will be deprecated in future versions.\n"
        "Please start using UIJson class instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if input_file.ui_json is None:
        raise GeoAppsError("The application needs a valid 'ui_json' file.")

    return UIJson.from_dict(input_file.ui_json)


class Driver(ABC):
    """
    # todo: Get rid of BaseParams to have a more robust DriverClass

    Base driver class.

    :param params: Application parameters.
    """

    _params_class: type[Options]

    def __init__(self, params: Options):
        self._out_group: UIJsonGroup | None = None
        self.params = params

    @property
    def params(self):
        """Application parameters."""
        return self._params

    @params.setter
    def params(self, val: Options):
        if not isinstance(val, self._params_class):
            raise TypeError(
                f"Parameters must be of type {self._params_class}.\n"
                f"Got {type(val)} instead."
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

    @classmethod
    def start(
        cls, filepath: str | Path | InputFile | UIJson, mode="r+", **kwargs
    ) -> Self:
        """
        Run application specified by 'filepath' ui.json file.

        :param filepath: Path to valid ui.json file for the application driver.
        :param mode: Mode to open the geoh5 file with.
        :param kwargs: Additional keyword arguments for Options class.

        :return: Self object.
        """

        if isinstance(filepath, InputFile):
            filepath = input_file_deprecation_warning(filepath)

        ifile = UIJson.read(filepath) if isinstance(filepath, str | Path) else filepath

        if not isinstance(ifile, UIJson):
            raise TypeError("Input file must be a string path or an InputFile object.")

        if ifile.geoh5 is None:
            raise GeoAppsError("The application needs a valid 'geoh5' file.")

        with Workspace(ifile.geoh5, mode=mode) as workspace:
            try:
                params = cls._params_class.build(ifile, workspace=workspace, **kwargs)
                logger.info("Initializing application . . .")
                driver = cls(params)
                logger.info("Running application . . .")
                driver.run()
                logger.info("Results saved to %s", params.geoh5.h5file)
            except GeoAppsError as error:
                logger.warning("\n\nApplicationError: %s\n\n", error)
                sys.exit(1)

        return driver

    def add_ui_json(self, entity: ObjectBase):
        """
        Add ui.json as FileData to entity.

        :param entity: Object to add ui.json file to.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = self.params.ui_json.write(Path(tmpdirname) / "temp.ui.json")
            entity.add_file(path)

    def update_monitoring_directory(
        self, entity: ObjectBase, copy_children: bool = True
    ):
        """
        If monitoring directory is active, copy entity to monitoring directory.

        :param entity: Object being added to monitoring directory.
        :param copy_children: If True, copy all children of the entity to the monitoring directory.
        """
        self.add_ui_json(entity)
        if (
            self.params.monitoring_directory is not None
            and Path(self.params.monitoring_directory).is_dir()
        ):
            monitored_directory_copy(
                str(Path(self.params.monitoring_directory).resolve()),
                entity,
                copy_children=copy_children,
            )

    @classmethod
    def get_default_ui_json_path(cls) -> Path | None:
        """
        Get the default ui.json file path for the application.

        :return: Path to default ui.json file.
        """
        if issubclass(cls._params_class, Options):
            return cls._params_class.default_ui_json
        return None

    @classmethod
    def get_default_ui_json(cls) -> UIJson:
        """
        Load the driver's default ui.json template from disk
        with no parameters filled in.

        :return: The default ui.json configuration.
        """
        if issubclass(cls._params_class, Options):
            return cls._params_class.get_default_ui_json()

        raise ValueError(f"Driver {cls} does not have a default ui.json.")


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
    default_ui_json: ClassVar[Path | None] = assets_path() / "uijson/base.ui.json"

    title: str = "Base Data"
    run_command: str = "geoapps_utils.base"
    conda_environment: str | None = None
    geoh5: Workspace
    monitoring_directory: str | Path | None = None
    out_group: UIJsonGroup | None = None

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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

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
    def build(
        cls,
        input_data: InputFile | dict | None | UIJson = None,
        workspace: Workspace | None = None,
        **kwargs,
    ) -> Self:
        """
        Build a dataclass from a dictionary or UIJson.

        :param input_data: Dictionary of parameters and values.
        :param workspace: Workspace to use for building parameters.

        :return: Dataclass of application parameters.
        """
        if isinstance(input_data, InputFile):
            input_data = input_file_deprecation_warning(input_data)
        elif input_data is None:
            input_data = {}

        data = input_data
        if isinstance(input_data, UIJson):
            data = input_data.to_params(workspace)

        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary or UIJson.")

        data.update(kwargs)
        options = cls.collect_input_from_dict(cls, data)  # type: ignore
        try:
            out = cls(**options)
        except ValidationError as errors:
            summary = "\n - ".join(
                f"{'.'.join(str(loc) for loc in error['loc'])}: "
                f"{error['msg']} for value -> {error['input']}"
                for error in errors.errors()
            )

            raise GeoAppsError(
                f"Invalid input data for {cls.__name__}:\n - {summary}"
            ) from errors

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
    def input_file(self) -> UIJson:
        """Return the current parameter state as a UIJson."""

        warnings.warn(
            "InputFile property is deprecated and will be removed in future versions. "
            "Use `ui_json` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.ui_json

    def serialize(self, mode="python"):
        """Return a demoted uijson dictionary representation the params data."""
        serialized = self.ui_json.model_dump(
            exclude_unset=True, by_alias=True, mode=mode
        )
        return serialized

    def update_out_group_options(self):
        """
        Serialize current state and save to the out_group options.
        """
        if self.out_group is None:
            raise ValueError("No output group defined to save options.")

        with fetch_active_workspace(self.geoh5, mode="r+"):
            self.out_group.options = self.serialize(mode="json")
            self.out_group.metadata = None

    @property
    def ui_json(self) -> UIJson:
        """
        The parent UIJson object.
        """
        ui_json = self.get_default_ui_json()
        ui_json.set_values(**self.flatten())

        return ui_json

    @classmethod
    def get_default_ui_json(cls) -> UIJson:
        """
        Load the driver's default ui.json template from disk
        with no parameters filled in.

        :return: The default ui.json configuration.
        """
        if cls.default_ui_json is None or not cls.default_ui_json.exists():
            raise ValueError(f"Class '{cls}' does not have a default ui.json.")

        return UIJson.read(cls.default_ui_json)
