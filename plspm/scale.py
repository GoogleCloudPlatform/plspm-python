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

import plspm.util as util, pandas as pd
from enum import Enum


class _Numeric:

    def scale(self, lv: str, mv: str, Z: pd.DataFrame, weights) -> pd.DataFrame:
        return util.treat(weights.mv_grouped_by_lv(lv, mv)) * weights.correction()


class _Raw:

    def scale(self, lv: str, mv: str, Z: pd.DataFrame, weights) -> pd.DataFrame:
        return weights.mv_grouped_by_lv(lv, mv)


class _Ordinal:

    def scale(self, lv: str, mv: str, Z: pd.DataFrame, weights) -> pd.DataFrame:
        return util.treat(weights.mv_grouped_by_lv(lv, mv)) * weights.correction()


class _Nominal:

    def scale(self, lv: str, mv: str, Z: pd.DataFrame, weights) -> pd.DataFrame:
        to_quantify = pd.concat([Z.loc[:, lv], weights.mv_grouped_by_lv(lv, mv)], axis=1)
        quantified = to_quantify.groupby(mv)[lv].mean()
        x_quantified = weights.dummies(mv).dot(quantified).to_frame().rename(columns={0: mv})
        return util.treat(x_quantified) * weights.correction()


class Scale(Enum):
    NUM = _Numeric()
    RAW = _Raw()
    ORD = _Ordinal()
    NOM = _Nominal()
