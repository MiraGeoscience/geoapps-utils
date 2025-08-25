# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2022-2025 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


try:
    from las_geoh5._version import __version__, __version_tuple__
except ModuleNotFoundError:  # pragma: no cover
    from datetime import datetime

    __date_str = datetime.today().strftime("%Y%m%d")
    __version__ = "0.0.0.dev0+" + __date_str
    __version_tuple__ = (0, 0, 0, "dev0", __date_str)

from geoapps_utils.utils import (
    conversions,
    formatters,
    importing,
    iterables,
    locations,
    numerical,
    plotting,
    transformations,
    workspace,
)
from geoapps_utils.utils.importing import GeoAppsError
from geoapps_utils.utils.importing import assets_path as assets_path_impl
from geoapps_utils.utils.logger import get_logger


def assets_path() -> Path:
    """Return the path to the assets folder."""
    return assets_path_impl(__file__)


__all__ = [
    "assets_path",
    "conversions",
    "formatters",
    "importing",
    "iterables",
    "locations",
    "numerical",
    "plotting",
    "transformations",
    "workspace",
]
