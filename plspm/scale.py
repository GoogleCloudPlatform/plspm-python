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

import plspm.util as util, pandas as pd, numpy as np
from enum import Enum


class _Numeric:

    def scale(self, lv: str, mv: str, z_by_lv: pd.Series, weights) -> pd.DataFrame:
        data = weights.mv_grouped_by_lv(lv, mv)
        finite_elements = np.isfinite(data).sum()
        return util.treat_numpy(data) * np.sqrt(finite_elements / (finite_elements - 1))


class _Raw:

    def scale(self, lv: str, mv: str, z_by_lv: pd.Series, weights) -> pd.DataFrame:
        return weights.mv_grouped_by_lv(lv, mv)


class _Ordinal:

    def _quantify(self, dummies, z_by_lv):
        scaling = [0] * (len(dummies[0]))
        for n in range(len(dummies[0])):
            scaling[n] = np.sum(dummies[:, n] * z_by_lv)
            scaling[n] = scaling[n] / np.sum(dummies[:, n])
        return scaling

    def _ordinalize(self, scaling, dummies, z_by_lv, sign: int):
        while True:
            ncols = len(dummies[0])
            for n in range(len(dummies[0]) - 1):
                if np.sign(scaling[n] - scaling[n + 1]) == sign:
                    dummies[:, n + 1] = dummies[:, n] + dummies[:, n + 1]
                    dummies = np.delete(dummies, n, axis=1)
                    scaling = self._quantify(dummies, z_by_lv)
                    break
            if len(dummies[0]) == 1 or len(dummies[0]) == ncols:
                break
        x_new = np.dot(dummies, scaling)
        return x_new, np.var(x_new)

    def scale(self, lv: str, mv: str, z_by_lv: np.ndarray, weights) -> pd.DataFrame:
        z_by_lv = weights.get_Z_for_mode_b(lv, mv, z_by_lv)
        to_quantify = np.array([weights.mv_grouped_by_lv(lv, mv), z_by_lv])
        quantified = util.groupby_mean(to_quantify)
        x_quant_incr, var_incr = self._ordinalize(quantified[1, :], weights.dummies(mv).copy(), z_by_lv, 1)
        x_quant_decr, var_decr = self._ordinalize(quantified[1, :], weights.dummies(mv).copy(), z_by_lv, -1)
        x_quantified = -x_quant_decr if var_incr < var_decr else x_quant_incr
        scaled = util.treat_numpy(x_quantified) * weights.correction()
        return scaled


class _Nominal:

    def scale(self, lv: str, mv: str, z_by_lv: np.ndarray, weights) -> pd.DataFrame:
        z_by_lv = weights.get_Z_for_mode_b(lv, mv, z_by_lv)
        to_quantify = np.array([weights.mv_grouped_by_lv(lv, mv), z_by_lv])
        quantified = util.groupby_mean(to_quantify)
        x_quantified = weights.dummies(mv).dot(quantified[1, :])
        return util.treat_numpy(x_quantified) * weights.correction()


class Scale(Enum):
    """
    Used to specify the measurement type of a manifest variable when performing calculations with nonmetric data.

    * :attr:`RAW` for numerical variables that require no transformation;
    * :attr:`NUM` for numerical variables that are suitable for linear transformation;
    * :attr:`ORD` for ordinal variables that are suitable for monotonic transformation;
    * :attr:`NOM` for nominal variables that are suitable for non-monotonic transformation.
    """
    NUM = _Numeric()
    RAW = _Raw()
    ORD = _Ordinal()
    NOM = _Nominal()
