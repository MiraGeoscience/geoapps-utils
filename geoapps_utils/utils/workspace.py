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

import time
from pathlib import Path

from geoh5py.workspace import Workspace


def get_output_workspace(
    live_link, workpath: str | Path = Path(), name: str = "Temp.geoh5"
):
    """
    Create an active workspace with check for GA monitoring directory.

    :param live_link: Application live link status.
    :param workpath: Path to output workspace.
    :param name: Name of output workspace.

    :return: Updated live link value.
    """
    if Path(name).suffix != ".geoh5":
        name += ".geoh5"
    workspace = Workspace.create(Path(workpath) / name)
    workspace.close()
    new_live_link = False
    time.sleep(1)
    # Check if GA digested the file already
    if (isinstance(workspace.h5file, Path | str)) and not Path(
        workspace.h5file
    ).is_file():
        workpath = Path(workpath) / ".working"
        workpath.mkdir(parents=True, exist_ok=True)
        workspace = Workspace.create(workpath / name)
        workspace.close()
        new_live_link = True
        if not live_link:
            print(
                "ANALYST Pro active live link found. Switching to monitoring directory..."
            )
    elif live_link:
        print(
            "ANALYST Pro 'monitoring directory' inactive. Reverting to standalone mode..."
        )
    workspace.open()
    # return new live link
    return workspace, new_live_link
