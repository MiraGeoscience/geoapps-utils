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

from abc import ABC, abstractmethod
from pathlib import Path

from geoh5py import Workspace
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile, monitored_directory_copy

from geoapps_utils.driver.data import BaseData
from geoapps_utils.driver.params import BaseParams


class BaseDriver(ABC):
    """
    # todo: Get rid of BaseParams to have a more robust DriverClass

    Base driver class.

    :param params: Application parameters.
    """

    _params_class: type[BaseData] | type[BaseParams] = BaseParams
    _validations: dict | None = None

    def __init__(self, params: BaseParams | BaseData):
        self._workspace: Workspace | None = None
        self._out_group: str | None = None
        self.params = params

        if (
            hasattr(self.params, "out_group")
            and self.params.out_group is None
            and not issubclass(self._params_class, BaseData)
        ):
            self.params.out_group = self.out_group

    @property
    def out_group(self):
        """Output group."""
        return self._out_group

    @property
    def params(self):
        """Application parameters."""
        return self._params

    @params.setter
    def params(self, val: BaseParams | BaseData):
        if not isinstance(val, (BaseParams, BaseData)):
            raise TypeError(
                "Parameters must be of type BaseParams or BaseData,"
                f" get {type(val)} instead."
            )
        self._params = val

    @property
    def workspace(self):
        """Application workspace."""
        if self._workspace is None and self._params is not None:
            self._workspace = self._params.geoh5

        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        """Application workspace."""

        if not isinstance(workspace, Workspace):
            raise TypeError(
                "Input value for `workspace` must be of type geoh5py.Workspace."
            )

        self._workspace = workspace

    @property
    def params_class(self):
        """Default parameter class."""
        return self._params_class

    @abstractmethod
    def run(self):
        """Run the application."""
        raise NotImplementedError

    @classmethod
    def start(cls, filepath: str | Path, driver_class=None):
        """
        Run application specified by 'filepath' ui.json file.

        :param filepath: Path to valid ui.json file for the application driver.
        :param driver_class: Application driver class.
        """

        if driver_class is None:
            driver_class = cls

        print("Loading input file . . .")
        filepath = Path(filepath).resolve()
        ifile = InputFile.read_ui_json(filepath, validations=cls._validations)

        with ifile.geoh5.open(mode="r+"):
            params = driver_class._params_class.build(ifile)
            print("Initializing application . . .")
            driver = driver_class(params)

        print("Running application . . .")
        driver.run()
        print(f"Results saved to {params.geoh5.h5file}")

        return driver

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
