# Copyright (c) 2026 Mira Geoscience Ltd.
#
# Distributed under the terms of the MIT License (see LICENSE).

"""Validate project requirements against every Pixi solve without installing it."""

import json
import subprocess
import tomllib
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
CONDA_NAMES = {"matplotlib": "matplotlib-base"}


def validate_requirements(
    requirements: list[str], packages: list[dict], platform: str, requires_python: str
) -> list[str]:
    """Return missing/incompatible requirements using the target marker environment."""
    versions = {
        canonicalize_name(package["name"]): package["version"] for package in packages
    }
    python = versions["python"]
    windows = platform == "win-64"
    markers = default_environment() | {
        "python_version": ".".join(python.split(".")[:2]),
        "python_full_version": python,
        "sys_platform": "win32" if windows else "linux",
        "platform_system": "Windows" if windows else "Linux",
        "os_name": "nt" if windows else "posix",
        "platform_machine": "AMD64" if windows else "x86_64",
        "extra": "",
    }
    errors: list[str] = []
    if not SpecifierSet(requires_python).contains(python, prereleases=True):
        errors.append(f"Python {python} does not satisfy {requires_python}")
    for requirement in map(Requirement, requirements):
        if requirement.marker and not requirement.marker.evaluate(markers):
            continue
        name = canonicalize_name(requirement.name)
        name = CONDA_NAMES.get(name, name)
        version = versions.get(name)
        if version is None or not requirement.specifier.contains(
            version, prereleases=True
        ):
            errors.append(f"{requirement}: locked version is {version or 'missing'}")
    return errors


def group_requirements(groups: dict, name: str) -> list[str]:
    """Expand PEP 735 include-group entries."""
    requirements = []
    for entry in groups[name]:
        if isinstance(entry, str):
            requirements.append(entry)
        else:
            requirements.extend(group_requirements(groups, entry["include-group"]))
    return requirements


def pixi_json(*args: str):
    """Read lock/workspace metadata without installing or updating environments."""
    return json.loads(
        subprocess.check_output(
            ["pixi", *args, "--json"], cwd=ROOT, text=True, encoding="utf-8"
        )
    )


def main() -> None:
    """Check all declared environment/platform pairs, including runtime-only solves."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = project["dependency-groups"]
    errors: list[str] = []
    for environment in pixi_json("info")["environments_info"]:
        requirements = [
            *project["project"]["dependencies"],
            *project["build-system"]["requires"],
        ]
        for feature in environment["features"]:
            if feature in groups:
                requirements.extend(group_requirements(groups, feature))
        for platform in environment["platforms"]:
            packages = pixi_json(
                "list",
                "--frozen",
                "--no-install",
                "--environment",
                environment["name"],
                "--platform",
                platform,
            )
            failures = validate_requirements(
                requirements, packages, platform, project["project"]["requires-python"]
            )
            conda = {
                canonicalize_name(package["name"])
                for package in packages
                if package["kind"] == "conda"
            }
            for name in environment["dependencies"]:
                if canonicalize_name(name) not in conda:
                    failures.append(f"{name} must be provided by Conda")
            if any(
                canonicalize_name(package["name"]) == project["project"]["name"]
                for package in packages
            ):
                failures.append("The local project must not be a locked dependency")
            label = f"{environment['name']}/{platform}"
            errors.extend(f"{label}: {failure}" for failure in failures)
            print(f"{label}: {'FAILED' if failures else 'OK'}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
