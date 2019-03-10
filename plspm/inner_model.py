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


def _summary(regression):
    summary = pd.DataFrame(0, columns=['estimate', 'std error', 't', 'p>|t|'], index=regression.params.index)
    summary['estimate'] = regression.params
    summary['std error'] = regression.bse
    summary['t'] = regression.tvalues
    summary['p>|t|'] = regression.pvalues
    return summary


def _effects(path: pd.DataFrame):
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
                effect = pd.Series({"from": from_lv, "to": to_lv, "direct": path.loc[to_lv, from_lv],
                                    "indirect": indirect_paths.loc[to_lv, from_lv],
                                    "total": total_paths.loc[to_lv, from_lv]}, name=from_lv + " -> " + to_lv)
                effects = effects.append(effect)
    return effects


class InnerModel:
    """Internal class that calculates the attributes of the inner model. Use the methods :meth:`~plspm.Plspm.inner_model`, :meth:`~plspm.Plspm.path_coefficients`, and :meth:`~plspm.Plspm.effects` defined on :class:`~.plspm.Plspm` to retrieve the inner model characteristics."""
    def __init__(self, path: pd.DataFrame, scores: pd.DataFrame):
        self.__summaries = {}
        self.__r_squared = pd.Series(0, index=path.index, name="r_squared")
        self.__path_coefficients = pd.DataFrame(0, columns=path.columns, index=path.index)
        endogenous = path.sum(axis=1).astype(bool)
        self.__endogenous = list(endogenous[endogenous == True].index)
        for dv in self.__endogenous:
            ivs = path.loc[dv,][path.loc[dv,] == 1].index
            exogenous = sm.add_constant(scores.loc[:, ivs])
            regression = sm.OLS(scores.loc[:, dv], exogenous).fit()
            self.__path_coefficients.loc[dv, ivs] = regression.params
            self.__r_squared.loc[dv] = regression.rsquared
            self.__summaries[dv] = _summary(regression).drop("const")
        self.__effects = _effects(self.__path_coefficients)

    def path_coefficients(self) -> pd.DataFrame:
        """Internal method that returns the path coefficients of the inner model."""
        return self.__path_coefficients

    def r_squared(self) -> pd.Series:
        """Internal method that returns r squared for the latent variables."""
        return self.__r_squared

    def inner_model(self) -> dict:
        """Internal method that returns summaries of the characteristics of the inner model for each latent variable."""
        return self.__summaries

    def effects(self) -> pd.DataFrame:
        """Internal method that returns indirect, direct, and total effects for each path in the model."""
        return self.__effects

    def endogenous(self) -> list:
        """Internal method that returns a list of the endogenous latent variables."""
        return self.__endogenous
