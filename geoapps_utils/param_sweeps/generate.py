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
import re
from pathlib import Path

from geoh5py.ui_json import InputFile

from geoapps_utils.base import Options


def generate(
    worker_uijson: str,
    parameters: list[str] | None = None,
    update_values: dict | None = None,
):
    """
    Generate an *_sweep.ui.json file to sweep parameters of the driver associated with 'file'.

    :param worker_uijson: Name of .ui.json file used to generate the sweep file.
    :param parameters: Parameters to include in the _sweep.ui.json file
    :param update_values: Updates for sweep files parameters
    """

    file_path = Path(worker_uijson).resolve(strict=True)
    ifile = InputFile.read_ui_json(file_path)

    options = Options(geoh5=ifile.data["geoh5"])
    ui_json = options.serialize()

    sweepfile = InputFile(ui_json=ui_json, validate=False)

    if sweepfile.data is None or sweepfile.ui_json is None:
        raise ValueError("Sweep file data is empty.")

    sweepfile.data.update({"worker_uijson": str(worker_uijson)})
    if update_values:
        sweepfile.data.update(**update_values)

    for param, value in ifile.data.items():
        if parameters is not None and param not in parameters:
            continue

        if type(value) in [int, float]:
            forms = sweep_forms(param, value)
            sweepfile.ui_json.update(forms)

    dirpath = file_path.parent
    filename = file_path.name.removesuffix(".ui.json")
    filename = re.sub(r"\._sweep$", "", filename)
    filename = f"{filename}_sweep.ui.json"

    print(f"Writing sweep file to: {dirpath / filename}")
    sweepfile.write_ui_json(name=filename, path=dirpath)


def sweep_forms(param: str, value: int | float) -> dict:
    """
    Return a set of three ui.json entries for start, end and n (samples).

    :param param: Parameter name
    :param value: Parameter value
    """
    group = param.replace("_", " ").capitalize()
    forms = {
        f"{param}_start": {
            "main": True,
            "group": group,
            "label": "starting",
            "value": value,
        },
        f"{param}_end": {
            "main": True,
            "group": group,
            "optional": True,
            "enabled": False,
            "label": "ending",
            "value": value,
        },
        f"{param}_n": {
            "main": True,
            "group": group,
            "dependency": f"{param}_end",
            "dependencyType": "enabled",
            "enabled": False,
            "label": "number of samples",
            "value": 1,
        },
    }

    return forms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a sweep file.")
    parser.add_argument("file", help="File with ui.json format.")
    parser.add_argument(
        "--parameters",
        help="List of parameters to be included as sweep parameters.",
        nargs="+",
    )
    args = parser.parse_args()
    generate(args.file, args.parameters)
