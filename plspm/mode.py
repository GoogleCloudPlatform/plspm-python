#!/usr/bin/python3
#
# Copyright (C) 2019 Google Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np, pandas as pd, scipy.linalg as linalg, plspm.util as util
from enum import Enum
from typing import Tuple


class _ModeA:

    def outer_weights_metric(self, data: pd.DataFrame, Z: pd.DataFrame, lv: str, mvs: list) -> pd.DataFrame:
        return (1 / data.shape[0]) * Z.loc[:, [lv]].T.dot(data.loc[:, mvs]).T

    def outer_weights_nonmetric(self, mv_grouped_by_lv: list, mv_grouped_by_lv_missing: list, Z: np.ndarray, lv: str,
                                correction: float) -> Tuple[np.ndarray, np.ndarray]:
        if lv in mv_grouped_by_lv_missing:
            weights = np.nansum(mv_grouped_by_lv[lv] * Z[:, np.newaxis], axis=0)
            weights = weights / np.sum(np.power(mv_grouped_by_lv_missing[lv] * Z[:, np.newaxis], 2), axis=0)
            Y = np.nansum(np.transpose(mv_grouped_by_lv[lv]) * weights[:, np.newaxis], axis=0)
            Y = Y / np.sum(np.power(np.transpose(mv_grouped_by_lv_missing[lv]) * weights[:, np.newaxis], 2), axis=0)
        else:   
            weights = np.dot(np.transpose(mv_grouped_by_lv[lv]), Z) / np.power(Z, 2).sum()
            Y = np.dot(mv_grouped_by_lv[lv], weights)
        Y = util.treat_numpy(Y) * correction
        return weights, Y


class _ModeB:

    def outer_weights_metric(self, data: pd.DataFrame, Z: pd.DataFrame, lv: str, mvs: list) -> pd.DataFrame:
        w, _, _, _ = linalg.lstsq(data.loc[:, mvs], Z.loc[:, [lv]])
        return pd.DataFrame(w, columns=[lv], index=mvs)

    def outer_weights_nonmetric(self, mv_grouped_by_lv: list, mv_grouped_by_lv_missing: list, Z: pd.DataFrame, lv: str,
                                correction: float) -> Tuple[np.ndarray, np.ndarray]:
        if lv in mv_grouped_by_lv_missing:
            raise Exception("Missing nonmetric data is not supported in mode B. LV with missing data: " + lv)
        weights, _, _, _ = linalg.lstsq(mv_grouped_by_lv[lv], Z)
        Y = np.dot(mv_grouped_by_lv[lv], weights)
        Y = util.treat_numpy(Y) * correction
        return weights, Y


class Mode(Enum):
    """
    Specify whether a given latent variable is reflective (mode A) or formative (mode B) with respect to its manifest variables.
    """
    A = _ModeA()
    B = _ModeB()
