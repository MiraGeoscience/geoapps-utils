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

from logging import getLogger

from geoapps_utils.base import Options


logger = getLogger(__name__)


class BaseData(Options):
    """
    Deprecated base class for data handling in geoapps-utils.
    """

    def __init__(self, *args, **kwargs):
        logger.warning(
            "Class 'geoapps_utils.driver.data.BaseData' will be removed in future release.\n "
            "Use 'geoapps_utils.base.Options' instead."
        )
        super().__init__(*args, **kwargs)
