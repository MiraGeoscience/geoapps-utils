# Copyright (c) 2026 Mira Geoscience Ltd.
#
# Distributed under the terms of the MIT License (see LICENSE).

import pytest

from devtools.validate_pixi import group_requirements, validate_requirements


@pytest.mark.parametrize(
    "requirements,packages,platform,expected",
    [
        (
            ["geoh5py>=0.14.0b1,<0.15", "matplotlib>=3.10,<3.11"],
            [("geoh5py", "0.14.0b1"), ("matplotlib-base", "3.10.9")],
            "linux-64",
            [],
        ),
        (
            ["numpy>=2.4.2,<2.5", "scipy>=1.17,<1.18"],
            [("numpy", "2.3.0")],
            "win-64",
            [
                "numpy<2.5,>=2.4.2: locked version is 2.3.0",
                "scipy<1.18,>=1.17: locked version is missing",
            ],
        ),
        (
            ['numpy>=2.4; sys_platform == "win32"'],
            [],
            "linux-64",
            [],
        ),
        (
            ['numpy>=2.4; sys_platform == "win32"'],
            [],
            "win-64",
            ['numpy>=2.4; sys_platform == "win32": locked version is missing'],
        ),
    ],
)
def test_locked_requirements(requirements, packages, platform, expected):
    packages = [("python", "3.12.16"), *packages]
    assert (
        validate_requirements(
            requirements,
            [{"name": name, "version": version} for name, version in packages],
            platform,
            ">=3.12,<3.15",
        )
        == expected
    )


def test_incompatible_python():
    assert validate_requirements(
        [], [{"name": "python", "version": "3.11.14"}], "linux-64", ">=3.12,<3.15"
    ) == ["Python 3.11.14 does not satisfy >=3.12,<3.15"]


def test_dependency_group_includes():
    groups = {"dev": [{"include-group": "test"}, "Pygments"], "test": ["pytest"]}
    assert group_requirements(groups, "dev") == ["pytest", "Pygments"]
