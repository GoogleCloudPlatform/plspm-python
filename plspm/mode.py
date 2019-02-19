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

import numpy as np, pandas as pd, scipy.linalg as linalg


class _ModeA:

    def outer_weights_nonmetric(self, mv_grouped_by_lv: list, Z: pd.DataFrame, lv: str, correction: float):
        weights = (mv_grouped_by_lv[lv].transpose().dot(Z.loc[:, [lv]])) / np.power(Z.loc[:, [lv]], 2).sum()
        Y = mv_grouped_by_lv[lv].dot(weights)
        Y = Y * correction / Y.std()
        return weights, Y


class _ModeB:

    def outer_weights_nonmetric(self, mv_grouped_by_lv: list, Z: pd.DataFrame, lv: str, correction: float):
        w, _, _, _ = linalg.lstsq(mv_grouped_by_lv[lv], Z.loc[:, [lv]])
        weights = pd.DataFrame(w, columns=[lv], index=mv_grouped_by_lv[lv].columns.values)
        Y = mv_grouped_by_lv[lv].dot(weights)
        Y = Y * correction / Y.std()
        return weights, Y


A = _ModeA()
B = _ModeB()
