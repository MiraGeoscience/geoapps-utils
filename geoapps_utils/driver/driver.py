# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                          '
#                                                                                   '
#  This file is part of geoapps-utils package.                                      '
#                                                                                   '
#  geoapps-utils is distributed under the terms and conditions of the MIT License   '
#  (see LICENSE file at the root of this source code package).                      '
#                                                                                   '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from abc import ABC
from logging import getLogger

from geoapps_utils.base import Driver


logger = getLogger(__name__)


class BaseDriver(Driver, ABC):
    """
    Deprecated base driver class import.
    """

    def __init__(self, *args, **kwargs):
        logger.warning(
            "Class 'geoapps_utils.driver.driver.BaseDriver' will be removed in future release.\n "
            "Use 'geoapps_utils.base.Driver' instead."
        )
        super().__init__(*args, **kwargs)
