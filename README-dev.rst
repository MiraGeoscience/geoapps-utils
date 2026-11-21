Setup for development
=====================

.. contents:: Table of Contents
   :local:
   :depth: 2

Install `Pixi`_ **0.84.2** and external `uv <https://docs.astral.sh/uv/>`_.
Keep both commands on ``PATH``; uv is not installed by Pixi. Clone the repository,
then install and test the locked Python 3.12 environment::

    pixi install --locked -e py312
    pixi run --frozen -e py312 install
    pixi run --frozen -e py312 test

The shared lock covers Windows and Linux with Python 3.12, 3.13 and 3.14.
Use ``py313`` or ``py314`` to test another supported version, or run
``pixi run --frozen test-pyvers`` to test all three. The ``runtime`` environments
omit development tools. The ``install`` task installs this checkout in editable
mode without changing the locked dependencies, then checks compatibility with
``uv pip check``. Tests and linting use the source checkout directly.

The Poetry build backend and its requirements are resolved in an isolated build
environment using ``pyproject.toml``. The local package's dynamic version is
deliberately excluded from ``pixi.lock`` so commits and tags do not invalidate
the dependency lock.

Use the pinned Pixi version locally and in CI: older releases may not support
the lock format or compute the same lock hashes.

.. _Pixi: https://pixi.sh/
.. _prek: https://prek.j178.dev/
.. _Poetry: https://python-poetry.org/

Dependencies and sources
^^^^^^^^^^^^^^^^^^^^^^^^

``pixi.toml`` defines Python 3.12, 3.13 and 3.14 environments for Windows and Linux.
``default`` includes the Python 3.12 test, lint and interactive development tools.
``py312``, ``py313`` and ``py314`` are the test environments used by CI.
``runtime``, ``runtime-py313`` and ``runtime-py314`` omit the development tools
and share their dependency versions with the corresponding test environments.

Scientific dependencies and indirect dependencies such as ``defusedxml`` come
from Conda. The ``mira-pypi`` feature selects the published ``geoh5py``
prerelease from the public PyPI index; no private-index authentication is needed.
For development against the ``release/ga-4.9`` source, the corresponding Git
dependency is documented as a comment in ``pixi.toml``.

The lock source check derives the allowed PyPI package names from the
``mira-pypi`` feature and checks every locked environment and platform. Package
names are read from ``pixi.lock`` metadata, so this also works with Git sources.
Keep compatible bounds on indirect Conda dependencies: Pixi solves Conda before
PyPI and does not backtrack between them.

Run a task in a particular environment with ``-e``::

    pixi run --frozen -e py314 test
    pixi run --frozen lint geoapps_utils tests
    pixi run --frozen repl

Updating dependencies
^^^^^^^^^^^^^^^^^^^^^

Runtime package requirements are defined in ``[project.dependencies]`` in
``pyproject.toml``. Development requirements use PEP 735 dependency groups.
Match those constraints in the appropriate dependencies/features of ``pixi.toml``.
Indirect dependencies which must be supplied by Conda belong in ``pixi.toml``,
not in the package's published requirements.

After changing constraints, regenerate and validate the lock::

    pixi lock
    pixi install --locked -e py312
    pixi run --frozen -e py312 install
    pixi run --frozen test-pyvers

To update packages within existing constraints, use ``pixi update`` instead.
Commit both the manifest changes and ``pixi.lock``.

The ``install`` task validates the actual installed requirements with
``uv pip check`` after installing the local project with ``--no-deps``.
It is a separate setup/validation step, not a dependency of ``test``,
``test-local`` or ``lint``. CI runs it explicitly before testing in each
test environment. CI exercises
Python 3.12-3.14 on both Windows and Linux; it does not install every
runtime-only environment just to validate it.

The project still uses `Poetry`_ and poetry-dynamic-versioning as its build
backend, but Pixi manages development dependencies. Do not generate a
``poetry.lock`` or use Poetry to install the development environment.

Testing local dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

The tasks adapted from ``DEVOPS-693`` support testing sibling checkouts::

    $ pixi run --frozen test-local
    $ pixi run --frozen test-local "geoh5py"
    $ pixi run --frozen show-pip-source geoh5py

``test-local`` installs the specified sibling directories in editable mode
with ``--no-deps`` and checks installed requirements before running pytest.
This intentionally changes the installed packages without changing the lock.
Run ``pixi reinstall --frozen -e default geoh5py`` to restore the locked dependency.

Configure Git hooks
^^^^^^^^^^^^^^^^^^^

Install `prek`_ and configure the existing ``.pre-commit-config.yaml`` hooks::

    $ prek install
    $ prek install -t prepare-commit-msg -t commit-msg
    $ prek run --all-files

The commit-message hooks check the JIRA issue ID against the branch name.
The pylint hook runs in the frozen Pixi environment.
Run ``prek run`` to check only staged files.

Tests and coverage
^^^^^^^^^^^^^^^^^^

Tests are in ``tests``; test filenames end with ``_test.py`` and test functions
start with ``test_``. The ``test`` task runs pytest with coverage and writes
``coverage.xml``. For an HTML coverage report::

    $ pixi run --frozen install
    $ pixi run --frozen pytest --cov --cov-report=html tests

Open ``htmlcov/index.html`` to view the report.

Git LFS
^^^^^^^

If large files are needed, install `Git LFS`_ and run ``git lfs install``.
Add tracking patterns with ``git lfs track`` and commit ``.gitattributes``
along with the files.

.. _Git LFS: https://git-lfs.com/

IDE: PyCharm
^^^^^^^^^^^^

Exclude ``.pixi`` from indexing and configure an existing Python interpreter:
``.pixi/envs/default/python.exe`` on Windows, or
``.pixi/envs/default/bin/python`` on Linux.
Mark ``tests`` as a Test Sources Root and use pytest as the test runner.

Copyright
^^^^^^^^^
Copyright (c) 2022-2026 Mira Geoscience Ltd.
