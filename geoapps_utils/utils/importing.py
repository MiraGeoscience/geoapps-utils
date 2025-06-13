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

import warnings
from contextlib import contextmanager
from logging import getLogger
from pathlib import Path
from typing import Annotated

from geoh5py.groups import DrillholeGroup
from pydantic import BaseModel, BeforeValidator, ConfigDict


logger = getLogger(__name__)


class DrillholeGroupValue(BaseModel):
    """
    Base class for group values.

    :param group_value: The Group containing the value.
    :param value: The list of str values to extract.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    group_value: DrillholeGroup
    value: Annotated[
        list[str], BeforeValidator(lambda v: v if isinstance(v, list) else [v])
    ]


@contextmanager
def warn_module_not_found():
    """
    Context manager to suppress ModuleNotFoundError exceptions, and warn instead.

    After the exception is suppressed, execution proceeds with the next
    statement following the with statement.

         with warn_module_not_found():
             from ipywidgets import Widget
         # Execution still resumes here if ipywidget is not found
    """
    try:
        yield
    except ModuleNotFoundError as error:
        module_name = error.name
        err = (
            f"Module '{module_name}' is missing from the environment. "
            f"Consider installing with: 'conda install -c conda-forge {module_name}'"
        )
        warnings.warn(err)


def assets_path(module_path: Path | str) -> Path:
    """
    Return the path to the assets folder, for the given module.

    :param module_path: Path of the module expected to have an associated asset folder.
    """

    source = Path(module_path).resolve()
    if source.is_file():
        source = source.parent
    assets_folder_name = f"{source.name}-assets"
    assets_folder = source.parent / assets_folder_name
    if not assets_folder.is_dir():
        raise RuntimeError(f"Assets folder not found: {assets_folder}")

    return assets_folder


class GeoAppsError(Exception):
    """
    Base class for exceptions in this module.
    """
