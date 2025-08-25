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

from pathlib import Path

import pytest
import tomli as toml
import yaml
from jinja2 import Template
from packaging.version import InvalidVersion, Version

import geoapps_utils


def get_conda_recipe_version():
    path = Path(__file__).resolve().parents[1] / "recipe.yaml"

    with open(str(path), encoding="utf-8") as file:
        content = file.read()

    template = Template(content)
    rendered_yaml = template.render()

    recipe = yaml.safe_load(rendered_yaml)

    return recipe["context"]["version"]


def test_version_is_consistent():
    project_version = Version(geoapps_utils.__version__)
    conda_version = Version(get_conda_recipe_version())
    assert conda_version.base_version == project_version.base_version


def _can_import_version():
    try:
        import geoapps_utils._version

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    _can_import_version(),
    reason="geoapps_utils._version can be imported: package is built",
)
def test_fallback_version_is_zero():
    project_version = Version(geoapps_utils.__version__)
    fallback_version = Version("0.0.0.dev0")
    assert project_version.base_version == fallback_version.base_version
    assert project_version.pre is None
    assert project_version.post is None
    assert project_version.dev == fallback_version.dev


@pytest.mark.skipif(
    not _can_import_version(),
    reason="geoh5py._version cannot be imported: uses a fallback version",
)
def test_conda_version_is_consistent():
    project_version = Version(geoapps_utils.__version__)
    conda_version = Version(get_conda_recipe_version())

    assert conda_version.is_devrelease == project_version.is_devrelease
    assert conda_version.is_prerelease == project_version.is_prerelease
    assert conda_version.is_postrelease == project_version.is_postrelease
    assert conda_version == project_version


def test_conda_version_is_pep440():
    version = Version(get_conda_recipe_version())
    assert version is not None


def validate_version(version_str):
    try:
        version = Version(version_str)
        return (version.major, version.minor, version.micro, version.pre, version.post)
    except InvalidVersion:
        return None


def test_version_is_valid():
    assert validate_version(geoapps_utils.__version__) is not None
