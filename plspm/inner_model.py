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

import pandas as pd, statsmodels.api as sm


def summary(regression):
    summary = pd.DataFrame(0, columns=['estimate', 'std error', 't', 'p>|t|'], index=regression.params.index)
    summary['estimate'] = regression.params
    summary['std error'] = regression.bse
    summary['t'] = regression.tvalues
    summary['p>|t|'] = regression.pvalues
    return summary


def effects(path: pd.DataFrame):
    indirect_paths = pd.DataFrame(0, index=path.index, columns=path.columns)
    effects = pd.DataFrame(columns=["from", "to", "direct", "indirect", "total"])
    num_lvs = len(list(path))
    if (num_lvs == 2):
        total_paths = path
    else:
        path_effects = {}
        path_effects[0] = path
        for i in range(1, num_lvs):
            path_effects[i] = path_effects[i - 1].dot(path)
            indirect_paths = indirect_paths + path_effects[i]
        total_paths = path + indirect_paths
    for from_lv in list(path):
        for to_lv in list(path):
            if from_lv != to_lv and total_paths.loc[to_lv, from_lv] != 0:
                effects = effects.append({"from": from_lv, "to": to_lv,
                                "direct": path.loc[to_lv, from_lv], "indirect": indirect_paths.loc[to_lv, from_lv],
                                          "total": total_paths.loc[to_lv, from_lv]}, ignore_index=True)
    return effects

class InnerModel:
    def __init__(self, path: pd.DataFrame, scores: pd.DataFrame):
        self.__summaries = {}
        self.__r_squared = pd.Series(0, index=path.index, name="r_squared")
        self.__path_coefficients = pd.DataFrame(0, columns=path.columns, index=path.index)
        endogenous = path.sum(axis=1).astype(bool)
        num_endo = endogenous.sum()
        for aux in range(0, num_endo):
            dv = endogenous[endogenous == True].index[aux]
            ivs = path.loc[dv,][path.loc[dv,] == 1].index
            exogenous = sm.add_constant(scores.loc[:, ivs])
            regression = sm.OLS(scores.loc[:, dv], exogenous).fit()
            self.__path_coefficients.loc[dv, ivs] = regression.params
            self.__r_squared.loc[dv] = regression.rsquared
            self.__summaries[dv] = summary(regression)
        self.__effects = effects(self.__path_coefficients)

    def path_coefficients(self) -> pd.DataFrame:
        return self.__path_coefficients

    def r_squared(self) -> pd.Series:
        return self.__r_squared

    def inner_model(self) -> dict:
        return self.__summaries

    def effects(self) -> pd.DataFrame:
        return self.__effects
