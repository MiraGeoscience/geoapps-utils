Setup for development
=====================

.. contents:: Table of Contents
   :local:
   :depth: 2

Prerequisites
^^^^^^^^^^^^^

* **Pixi and uv:** Install `Pixi <https://pixi.sh/>`_ 0.84.2 and external
  `uv <https://docs.astral.sh/uv/>`_. Keep both commands on ``PATH``; uv is not
  installed by Pixi.
* **Git LFS:** This repository uses Git LFS. Follow GitHub's `installation
  instructions
  <https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage>`_.
* **prek:** Install `prek <https://prek.j178.dev/>`_ for the configured Git
  hooks.
* **IDE interpreter:** Configure your IDE to use the interpreter inside
  ``.pixi/envs/default`` and exclude ``.pixi`` from source indexing.
  For **PyCharm**, follow Pixi's `JetBrains integration guide
  <https://pixi.prefix.dev/latest/integration/editor/jetbrains/>`_.
* **JFrog authentication:** Pixi uses uv's PyPI authentication, which is
  separate from ``pixi auth login``. Configure a protected ``.netrc`` file
  (``%HOME%\_netrc`` on Windows) with your Artifactory machine, login, and
  password or access token, as described in `Pixi's authentication guide
  <https://pixi.prefix.dev/latest/deployment/authentication/#netrc-file>`_.
  Never add credentials to ``pixi.toml`` or ``pixi.lock``.

.. note:: External contributors using Pixi

   If you do not have Mira Artifactory credentials, comment out ``index-url``
   under ``[feature.mira-pypi.pypi-options]`` in ``pixi.toml``, then run
   ``pixi lock``. Packages configured with version references will resolve from
   public PyPI if the required versions are available there; packages configured
   with Git references will resolve from Git.
   Keep these local ``pixi.toml`` and ``pixi.lock`` changes uncommitted.

Install and test
^^^^^^^^^^^^^^^^

The shared lock covers Windows and Linux with multiple Python versions::

    pixi install --locked -e py312
    pixi run --frozen -e py312 install
    pixi run --frozen -e py312 test

The ``install`` task installs this checkout in editable mode without changing
locked dependencies, then checks package compatibility.

The ``runtime`` environment omits test and lint tools.

Working with local sibling checkouts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a one-off test run, ``test-local`` installs the sibling checkouts as
editable dependencies and then runs the tests::

    pixi run -e py312 test-local

For ongoing development or IDE debugging, install editable sibling checkouts
into the default environment (the IDE environment) with::

    pixi run use-local-deps

Pixi syncs the environment from the manifest and lock file before running the
task; the task then installs the local checkouts. A subsequent ``pixi run``
without ``--as-is`` syncs again and replaces those editable installs with the
locked packages. An IDE configured to launch the ``default`` environment's
Python interpreter directly does not run ``pixi`` before each launch, so the
editable installs remain in effect.

Building documentation
^^^^^^^^^^^^^^^^^^^^^^

This repository has no ``docs`` environment or ``build-docs`` task.

Updating dependencies
^^^^^^^^^^^^^^^^^^^^^

Keep published package requirements in ``pyproject.toml`` and their Conda/PyPI
environment equivalents in ``pixi.toml`` aligned. Update the recipe's matching
runtime requirements when a dependency constraint changes.

After editing requirements, run ``pixi update`` to update dependencies, or
``pixi lock`` to resolve the manifest and refresh the lock file without
installing the environments.

Commit ``pixi.lock`` with the manifest changes. In ``pixi.toml``, the
``mira-pypi`` feature lists Mira packages resolved from PyPI or GitHub:

- On a development branch, comment out published versions and use git sources.
- When preparing a release, use published versions and comment out git sources.

The lock file is checked by ``tests/lock_test.py``: it ensures all dependencies
are resolved from Conda, except for the Mira packages listed under the
``mira-pypi`` feature. Keep compatibility bounds on indirect Conda dependencies:
Pixi solves Conda before PyPI and does not backtrack between them.

Git hooks
^^^^^^^^^

Install `prek <https://prek.j178.dev/>`_ and use the existing hook configuration::

    prek install -t prepare-commit-msg -t commit-msg -t pre-commit -f --prepare-hooks
    prek run --all-files

The local pylint hook runs in the locked ``py312`` environment.

Test layout
^^^^^^^^^^^

Tests live under ``tests`` and use the ``_test.py`` filename suffix.

---------

*Copyright (c) 2023-2026 Mira Geoscience Ltd.*
