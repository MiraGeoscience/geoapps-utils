#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np

from geoapps_utils.utils.plotting import inv_symlog, symlog


def test_plotting_symlog():
    thresh = 1.0
    vals = np.logspace(-6, 6, 13)
    symlog_vals = symlog(vals, thresh)
    inv_symlog_vals = inv_symlog(symlog_vals, thresh)
    thresh_vals = symlog_vals[symlog_vals > thresh]

    assert np.allclose(vals, inv_symlog_vals)
    assert len(thresh_vals) == 6
    assert np.all(np.diff(thresh_vals) > 0.9)
