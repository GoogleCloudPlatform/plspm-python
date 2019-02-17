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

    def update_outer_weights(self, mv_grouped_by_lv, weights, Y, Z, lv, correction):
        weights[lv] = (mv_grouped_by_lv[lv].transpose().dot(Z.loc[:, [lv]])) / np.power(Z.loc[:, [lv]], 2).sum()
        Y.loc[:, [lv]] = mv_grouped_by_lv[lv].dot(weights[lv])
        Y.loc[:, [lv]] = Y.loc[:, [lv]] * correction / Y.loc[:, [lv]].std()


class _ModeB:

    def update_outer_weights(self, mv_grouped_by_lv, weights, Y, Z, lv, correction):
        w, _, _, _ = linalg.lstsq(mv_grouped_by_lv[lv], Z.loc[:, [lv]])
        weights[lv] = pd.DataFrame(w, columns=[lv], index=mv_grouped_by_lv[lv].columns.values)
        Y.loc[:, [lv]] = mv_grouped_by_lv[lv].dot(weights[lv])
        Y.loc[:, [lv]] = Y.loc[:, [lv]] * correction / Y.loc[:, [lv]].std()


A = _ModeA()
B = _ModeB()
