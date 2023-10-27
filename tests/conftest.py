#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest
import tomli as toml


@pytest.fixture
def pyproject() -> dict:
    """Return the pyproject.toml as a dictionary."""

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject_path, "rb") as pyproject_file:
        return toml.load(pyproject_file)
