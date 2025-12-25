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

import argparse
import itertools
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from pydantic import ConfigDict

from geoapps_utils.base import Driver, Options
from geoapps_utils.run import fetch_driver_class


class SweepParams(Options):
    """Parametrizes a sweep of the worker driver."""

    model_config = ConfigDict(frozen=False, extra="allow")

    title: str = "Parameter sweep"
    run_command: str = "param_sweeps.driver"

    worker_uijson: Path

    def worker_parameters(self) -> list[str]:
        """Return all sweep parameter names."""
        if self.model_extra is None:
            return []

        return [
            k.replace("_start", "") for k in self.model_extra if k.endswith("_start")
        ]

    def parameter_sets(self) -> dict:
        """Return sets of parameter values that will be combined to form the sweep."""

        names = self.worker_parameters()

        sets = {}
        for name in names:
            sweep = (
                getattr(self, f"{name}_start"),
                getattr(self, f"{name}_end"),
                getattr(self, f"{name}_n"),
            )
            if sweep[1] is None:
                sets[name] = [sweep[0]]
            else:
                sets[name] = [type(sweep[0])(s) for s in np.linspace(*sweep)]

        return sets


class SweepDriver(Driver):
    """Sweeps parameters of a worker driver."""

    _params_class = SweepParams

    def __init__(self, params: SweepParams):
        super().__init__(params)
        self.working_directory = str(Path(self.params.geoh5.h5file).parent)

    @staticmethod
    def uuid_from_params(params: tuple) -> str:
        """
        Create a deterministic uuid.

        :param params: Tuple containing the values of a sweep iteration.

        :returns: Unique but recoverable uuid file identifier string.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(hash(params))))

    def get_lookup(self) -> dict:
        """Generate lookup table for sweep trials."""

        lookup = {}
        sets = self.params.parameter_sets()
        iterations = list(itertools.product(*sets.values()))
        for iteration in iterations:
            param_uuid = SweepDriver.uuid_from_params(iteration)
            lookup[param_uuid] = dict(zip(sets.keys(), iteration, strict=False))
            lookup[param_uuid]["status"] = "pending"

        lookup = self.update_lookup(lookup, gather_first=True)
        return lookup

    def update_lookup(self, lookup: dict, gather_first: bool = False) -> dict:
        """Updates lookup with new entries. Ensures any previous runs are incorporated."""
        lookup_path = Path(self.working_directory) / "lookup.json"
        if lookup_path.is_file() and gather_first:  # In case restarting
            with open(lookup_path, encoding="utf8") as file:
                lookup.update(json.load(file))

        with open(lookup_path, "w", encoding="utf8") as file:
            json.dump(lookup, file, indent=4)

        return lookup

    def write_files(self, lookup):
        """Write ui.geoh5 and ui.json files for sweep trials."""
        ifile = InputFile.read_ui_json(self.params.worker_uijson)
        for name, trial in lookup.items():
            if trial["status"] != "pending":
                continue

            iter_h5file = str(Path(self.workspace.h5file).parent / f"{name}.ui.geoh5")
            shutil.copy(self.workspace.h5file, iter_h5file)

            ifile.update_ui_values(
                dict(
                    {key: val for key, val in trial.items() if key != "status"},
                    **{"geoh5": Workspace(iter_h5file)},
                )
            )

            ifile.name = f"{name}.ui.json"
            ifile.path = str(Path(self.workspace.h5file).parent)
            ifile.write_ui_json()
            lookup[name]["status"] = "written"

        _ = self.update_lookup(lookup)

    def run(self):
        """Execute a sweep."""

        lookup = self.get_lookup()
        self.write_files(lookup)

        for name, trial in lookup.items():
            file_path = Path(self.working_directory) / f"{name}.ui.json"
            if trial["status"] == "complete":
                continue

            trial["status"] = "processing"
            self.update_lookup(lookup)

            driver = fetch_driver_class(file_path)
            driver.start(file_path)

            trial["status"] = "complete"
            self.update_lookup(lookup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run parameter sweep of worker driver."
    )
    parser.add_argument("file", help="File with ui.json format.")

    args = parser.parse_args()
    SweepDriver.start(Path(args.file).resolve(strict=True), mode="r")
