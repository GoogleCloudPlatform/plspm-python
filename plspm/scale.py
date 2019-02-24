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
        return util.treat(weights.mv_grouped_by_lv(lv, mv)) * weights.correction()


class _Raw:

    def scale(self, lv: str, mv: str, z_by_lv: pd.Series, weights) -> pd.DataFrame:
        return weights.mv_grouped_by_lv(lv, mv)


class _Ordinal:

    def _quantify(self, dummies: pd.DataFrame, z_by_lv: pd.Series):
        scaling = pd.Series(0, dummies.columns)
        for col in dummies:
            scaling.loc[col] = (dummies.loc[:,col] * z_by_lv).sum()
            scaling.loc[col] = scaling.loc[col] / dummies.loc[:,col].sum()
        return scaling

    def _ordinalize(self, scaling: pd.Series, dummies: pd.DataFrame, z_by_lv: pd.Series, sign: int):
        # This is abysmally slow
        while True:
            ncols = len(dummies.columns)
            for n in range(len(dummies.columns) - 1):
                if np.sign(scaling.iloc[n] - scaling.iloc[n+1]) == sign:
                    dummies.iloc[:,n+1] = dummies.iloc[:,n] + dummies.iloc[:,n+1]
                    dummies = dummies.drop(dummies.columns[n], axis=1)
                    scaling = self._quantify(dummies, z_by_lv)
                    break
            if len(dummies.columns) == 1 or len(dummies.columns) == ncols:
                break
        x_new = dummies.dot(scaling)
        return x_new, x_new.var()

    def scale(self, lv: str, mv: str, z_by_lv: pd.Series, weights) -> pd.DataFrame:
        z_by_lv = weights.get_Z_for_mode_b(lv, mv, z_by_lv)
        to_quantify = pd.concat([z_by_lv, weights.mv_grouped_by_lv(lv, mv)], axis=1)
        quantified = to_quantify.groupby(mv)[lv].mean()
        x_quant_incr, var_incr = self._ordinalize(quantified.copy(), weights.dummies(mv).copy(), z_by_lv, 1)
        x_quant_decr, var_decr = self._ordinalize(quantified.copy(), weights.dummies(mv).copy(), z_by_lv, -1)
        x_quantified = -x_quant_decr if var_incr < var_decr else x_quant_incr
        scaled = util.treat(x_quantified.to_frame().rename(columns={0: mv})) * weights.correction()
        return scaled

class _Nominal:

    def scale(self, lv: str, mv: str, z_by_lv: pd.Series, weights) -> pd.DataFrame:
        z_by_lv = weights.get_Z_for_mode_b(lv, mv, z_by_lv)
        to_quantify = pd.concat([z_by_lv, weights.mv_grouped_by_lv(lv, mv)], axis=1)
        quantified = to_quantify.groupby(mv)[lv].mean()
        x_quantified = weights.dummies(mv).dot(quantified).to_frame().rename(columns={0: mv})
        return util.treat(x_quantified) * weights.correction()


class Scale(Enum):
    NUM = _Numeric()
    RAW = _Raw()
    ORD = _Ordinal()
    NOM = _Nominal()
