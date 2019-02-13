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

import statsmodels.formula.api as sm, numpy as np


class _CentroidInnerWeightCalculator:

    def calculate(self, path, y):
        return y.corr().mul(path + path.transpose()).apply(lambda x: np.sign(x))

class _FactorialInnerWeightCalculator:

    def calculate(self, path, y):
        return y.cov().mul(path + path.transpose())

class _PathInnerWeightCalculator:

    def calculate(self, path, y):
        E = path.copy()
        for column in list(E):
            follow = path.loc[column,:] == 1
            if path.loc[column,:].sum() > 0:
                E.loc[follow,column] = sm.OLS(y.loc[:,column], y.loc[:,follow]).fit().params
            predec = path.loc[:,column] == 1
            if path.loc[:,column].sum() > 0:
                E.loc[predec,column] = y.loc[:,predec].corrwith(y.loc[:,column])
        return E

CENTROID = _CentroidInnerWeightCalculator()
PATH = _PathInnerWeightCalculator()
FACTORIAL = _FactorialInnerWeightCalculator()
