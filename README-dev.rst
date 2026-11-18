Setup for development
=====================

.. contents:: Table of Contents
   :local:
   :depth: 2

Install `Pixi`_ **0.67.2**, clone the Git repository, then run::

    $ pixi lock --check
    $ pixi run --frozen install
    $ pixi run --frozen test

Pixi installs the locked development dependencies in ``.pixi/envs/default``.
The local package is installed in editable mode, without resolving or replacing
its runtime dependencies. The Poetry build backend and its requirements are
resolved in an isolated build environment using ``pyproject.toml``.
The local package's dynamic version is deliberately excluded from ``pixi.lock``
so commits and tags do not invalidate the dependency lock.

.. _Pixi: https://pixi.sh/
.. _prek: https://prek.j178.dev/
.. _Poetry: https://python-poetry.org/

Environments and tasks
^^^^^^^^^^^^^^^^^^^^^^

``pixi.toml`` defines Python 3.12, 3.13 and 3.14 environments for Windows and Linux.
``default`` includes the Python 3.12 test, lint and interactive development tools.
``py312``, ``py313`` and ``py314`` are the test environments used by CI.
``runtime``, ``runtime-py313`` and ``runtime-py314`` omit the development tools
and share their dependency versions with the corresponding test environments.

Scientific dependencies and indirect dependencies such as ``defusedxml`` come
from Conda. The published ``geoh5py`` prerelease comes from PyPI.

Run a task in a particular environment with ``-e``::

    $ pixi run --frozen -e py314 test
    $ pixi run --frozen lint geoapps_utils tests
    $ pixi run --frozen repl

To run tests across all supported Python versions::

    $ pixi run --frozen test-pyvers

To enter an interactive development shell::

    $ pixi shell --frozen

Updating dependencies
^^^^^^^^^^^^^^^^^^^^^

Runtime package requirements are defined in ``[project.dependencies]`` in
``pyproject.toml``. Development requirements use PEP 735 dependency groups.
Match those constraints in the appropriate dependencies/features of ``pixi.toml``.
Indirect dependencies which must be supplied by Conda belong in ``pixi.toml``,
not in the package's published requirements.

After changing constraints, regenerate and validate the lock::

    $ pixi lock
    $ pixi lock --check
    $ pixi run --frozen test-pyvers

To update packages within existing constraints, use ``pixi update`` instead.
Commit both the manifest changes and ``pixi.lock``.

The ``install`` task validates the actual installed requirements with
``uv pip check`` after installing the local project with ``--no-deps``.
The ``test`` and ``lint`` tasks depend on ``install``, so this check runs in
each installed test environment before testing or linting. CI exercises
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
