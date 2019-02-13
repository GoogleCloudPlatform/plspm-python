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

import plspm.util as util, pandas as pd, numpy as np, statsmodels.api as sm

def summary(regression):
    summary = pd.DataFrame(0, columns=['estimate', 'std error', 't', 'p>|t|'], index=regression.params.index)
    summary['estimate'] = regression.params
    summary['std error'] = regression.bse
    summary['t'] = regression.tvalues
    summary['p>|t|'] = regression.pvalues
    return summary

class InnerModel:

    def __init__(self, path, y):
        self.__summaries = {}
        self.__r_squared = pd.Series(0, index=path.index, name="r_squared")
        self.__path_coefficients = pd.DataFrame(0, columns=path.columns, index=path.index)
        endogenous = path.sum(axis=1).astype(bool)
        num_endo = endogenous.sum()
        for aux in range(0, num_endo):
            dv = endogenous[endogenous == True].index[aux]
            ivs = path.loc[dv,][path.loc[dv,] == 1].index
            exogenous = sm.add_constant(y.loc[:, ivs])
            regression = sm.OLS(y.loc[:, dv], exogenous).fit()
            self.__path_coefficients.loc[dv, ivs] = regression.params
            self.__r_squared.loc[dv] = regression.rsquared
            self.__summaries[dv] = summary(regression)

    def path_coefficients(self):
        return self.__path_coefficients

    def r_squared(self):
        return self.__r_squared

    def inner_model(self):
        return self.__summaries
