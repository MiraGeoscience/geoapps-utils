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

import importlib
import json
import logging
import re
from pathlib import Path

import pytest
import yaml
from geoh5py.ui_json import UIJson
from packaging.version import Version

import geoapps_utils
from geoapps_utils.base import Options


def _get_json_version() -> str:
    version_json_path = Path(__file__).resolve().parents[1] / "_version.json"
    with version_json_path.open(encoding="utf-8") as file:
        version_json = json.load(file)
    return version_json["version"]


def _get_conda_recipe_version_def() -> str:
    recipe_path = Path(__file__).resolve().parents[1] / "recipe.yaml"

    with recipe_path.open(encoding="utf-8") as file:
        recipe = yaml.safe_load(file)
    return recipe["context"]["version"]


def _version_module_exists():
    try:
        importlib.import_module("geoapps_utils._version")
        return True
    except ModuleNotFoundError:
        return False


def test_conda_recipe_version_loads_json():
    conda_version_def = _get_conda_recipe_version_def()
    regex = (
        r"\$\{\{\s*load_from_file\(\s*['\"](_version\.json)['\"]\s*\)"
        r"\s*\.version\b.*\}\}"
    )
    regex_match = re.match(regex, conda_version_def)
    assert regex_match is not None


@pytest.mark.skipif(
    _version_module_exists(),
    reason="geoapps_utils._version can be found: package is built",
)
def test_fallback_version_is_zero():
    project_version = Version(geoapps_utils.__version__)
    fallback_version = Version("0.0.0.dev0")
    assert project_version.base_version == fallback_version.base_version
    assert project_version.pre is None
    assert project_version.post is None
    assert project_version.dev == fallback_version.dev


@pytest.mark.skipif(
    not _version_module_exists(),
    reason="geoapps_utils._version cannot be found: uses a fallback version",
)
def test_version_json_is_consistent():
    project_version = Version(geoapps_utils.__version__)
    json_version = Version(_get_json_version())
    assert project_version == json_version


@pytest.mark.parametrize(
    "version_input,expected",
    [
        # Normal version
        ("1.2.3", "1.2.3"),
        # Dev version
        ("1.2.3.dev1", "1.2.3"),
        # Post-release version
        ("1.2.3.post1", "1.2.3"),
        # RC pre-release version
        ("1.2.3rc1", "1.2.3"),
        # Alpha pre-release version (should not normalize)
        ("1.2.3a1", "1.2.3a1"),
        # Beta pre-release version (should not normalize)
        ("1.2.3b1", "1.2.3b1"),
        # Local version
        ("1.2.3+local", "1.2.3"),
        # Combined cases with RC and post
        ("1.2.3rc1.post2+local", "1.2.3"),
        # Combined cases with RC and dev
        ("1.2.3rc1.dev2+local", "1.2.3"),
        # Combined cases with pre non-RC and post
        ("1.2.3b1.post2+local", "1.2.3b1"),
        # Combined cases with pre non-RC and dev
        ("1.2.3b1.dev2+local", "1.2.3b1"),
    ],
)
def test_comparable_version(version_input, expected):
    """Test the comparable_version method of SimPEGDriversUIJson."""
    assert Options.comparable_version(version_input) == expected


@pytest.mark.parametrize(
    "version_input,package_version,should_warn",
    [
        # Different version (should warn)
        ("1.0.0", "2.0.0", True),
        # Same version (should not warn)
        ("2.0.0", "2.0.0", False),
        # Post-release variant (should not warn)
        ("2.0.0.post1", "2.0.0", False),
        ("2.0.0", "2.0.0.post1", False),
        # RC variant (should not warn)
        ("2.0.0rc1", "2.0.0", False),
        ("2.0.0", "2.0.0rc1", False),
        ("2.0.0rc1", "2.0.0rc2", False),
        # differ by the pre-release number, non RC (should warn)
        ("2.0.0a1", "2.0.0a2", True),
        ("2.0.0b1", "2.0.0b2", True),
        ("2.0.0a1", "2.0.0", True),
        ("2.0.0", "2.0.0a1", True),
        ("2.0.0a1", "2.0.0b1", True),
        ("2.0.0b1", "2.0.0a1", True),
        ("2.0.0rc1", "2.0.0b1", True),
        ("2.0.0b1", "2.0.0rc1", True),
        # same normalized versions (should not warn)
        ("2.0.0-beta.1", "2.0.0b1", False),
        ("2.0.0b1", "2.0.0-beta.1", False),
    ],
)
def test_version_warning(
    monkeypatch,
    caplog,
    version_input,
    package_version,
    should_warn,
):
    """Test version warning behavior with mocked package version."""
    # Patch the global distribution finder function
    monkeypatch.setattr(
        importlib.metadata, "version", lambda package_name: package_version
    )

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        uijson = UIJson.read(geoapps_utils.assets_path() / "uijson" / "base.ui.json")
        uijson.version = version_input
        Options.build(uijson)

        warning_message = (
            f"version '{version_input}' does not match the "
            f"current 'geoapps_utils' version"
        )
        warning_found = any(
            warning_message in record.message for record in caplog.records
        )

        assert warning_found == should_warn
