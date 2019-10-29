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

import statsmodels.api as sm, numpy as np, pandas as pd
from enum import Enum


class _CentroidInnerWeightCalculator:

    def calculate(self, path: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return np.sign(np.corrcoef(y, rowvar=False) * (path + path.transpose()))


class _FactorialInnerWeightCalculator:

    def calculate(self, path: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return np.cov(y, rowvar=False) * (path + path.transpose())


class _PathInnerWeightCalculator:

    def calculate(self, path: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        E = path.values.astype(np.float64)
        for i in range(E.shape[0]):
            follow = path.iloc[i, :] == 1
            if path.iloc[i, :].sum() > 0:
                E[follow, i] = sm.OLS(y[:, i], y[:, follow]).fit().params
            predec = path.iloc[:, i] == 1
            if path.iloc[:, i].sum() > 0:
                E[predec, i] = np.corrcoef(np.column_stack((y[:, predec], y[:, i])), rowvar=False)[:,-1][:-1]
        return E


class Scheme(Enum):
    """
    The scheme to use to calculate inner weights.
    """
    CENTROID = _CentroidInnerWeightCalculator()
    PATH = _PathInnerWeightCalculator()
    FACTORIAL = _FactorialInnerWeightCalculator()
