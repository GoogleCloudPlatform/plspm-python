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
from typing import Tuple

import numpy as np, pandas as pd, plspm.config as c, statsmodels.api as sm, plspm.util as util, math
from plspm.scheme import Scheme
from plspm.mode import Mode

pd.options.mode.chained_assignment = None  # default='warn'


class _MetricWeights:
    """Internal class that calculates weights and scores when using metric data."""
    def __init__(self, data: pd.DataFrame, config: c.Config, correction: float):
        odm = config.odm()
        weight_factors = correction / data.dot(odm).std(axis=0)
        self.__mvs = list(odm.index)
        wf_diag = pd.DataFrame(np.diag(weight_factors), index=weight_factors.index, columns=weight_factors.index)
        weights = odm.dot(wf_diag)
        self.__w_old = weights.sum(axis=1).to_frame(name="weight")
        self.__data = data
        self.__config = config
        self.__weights = weights
        self.__correction = correction

    def iterate(self, inner_weight_calculator: Scheme) -> float:
        lvs = self.__config.lvs()
        y = self.__data.dot(self.__weights).reindex(lvs, axis=1)
        y = util.treat(y) / self.__correction
        inner_weights = pd.DataFrame(inner_weight_calculator.value.calculate(self.__config.path(), y.values), index=lvs, columns=lvs)
        Z = y.dot(inner_weights)
        for lv in list(lvs):
            mvs = self.__config.mvs(lv)
            weights = self.__config.mode(lv).value.outer_weights_metric(self.__data, Z, lv, mvs)
            self.__weights.loc[mvs, [lv]] = weights
        w_new = self.__weights.sum(axis=1).to_frame(name="weight")
        convergence = np.power(self.__w_old.abs() - w_new.abs(), 2).sum(axis=1).sum(axis=0)
        self.__w_old = w_new
        return convergence

    def calculate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        weight_factors = 1 / (self.__data.dot(self.__weights).std(axis=0) / self.__correction)
        wf_diag = pd.DataFrame(np.diag(weight_factors), index=weight_factors.index, columns=weight_factors.index)
        weights = self.__weights.dot(wf_diag)
        y = self.__data.dot(weights)
        cor = pd.concat([self.__data, y], axis=1).corr().loc[list(self.__data), list(y)]
        odm = weights.apply(lambda x: x!= 0).astype(int)
        sign = lambda x : math.copysign(1.0, x)
        w_sign = (cor * odm).applymap(sign).sum(axis=0).apply(sign)
        if -1 in w_sign.values:
            w_sign = w_sign.apply(lambda x : -1 if x == 0 else x)
            w_sign_matrix = pd.DataFrame(np.diag(w_sign), index=w_sign.index, columns=w_sign.index)
            y = y.dot(w_sign_matrix)
        weights = pd.DataFrame(weights.sum(axis=1), index=self.__mvs, columns=["weight"])
        return self.__data, y, weights


class _NonmetricWeights:
    """Internal class that calculates weights and scores when using nonmetric data."""
    def __init__(self, data: pd.DataFrame, config: c.Config, correction: float):
        self.__mv_grouped_by_lv_initial = {}
        self.__mvs = []
        mv_grouped_by_lv = {}
        self.__mv_grouped_by_lv_missing = {}
        y = np.zeros((len(data.index), len(config.lvs())), dtype=np.float64)
        for i, lv in enumerate(config.lvs()):
            mvs = config.mvs(lv)
            self.__mvs.extend(mvs)
            mv_grouped_by_lv[lv] = data.filter(config.mvs(lv)).values.astype(np.float64)
            self.__mv_grouped_by_lv_initial[lv] = mv_grouped_by_lv[lv].copy()
            sizes = mv_grouped_by_lv[lv].shape[1]
            weight = [1 / np.sqrt(sizes)] * sizes
            if np.isnan(np.sum(mv_grouped_by_lv[lv])):
                self.__mv_grouped_by_lv_missing[lv] = 1 - np.isnan(mv_grouped_by_lv[lv])
                for j in range(len(data.index)):
                    numerator = np.nansum(mv_grouped_by_lv[lv][j, :] * weight)
                    denominator = np.power(weight * self.__mv_grouped_by_lv_missing[lv][j, :], 2).sum()
                    if denominator == 0:
                        raise ValueError("All mvs for lv " + lv + " in row " + str(j) + " are NaN.")
                    y[j, i] = numerator / denominator
            else:
                y[:, i] = np.dot(mv_grouped_by_lv[lv], weight)
        self.__weights = {}
        self.__y = y
        self.__mv_grouped_by_lv = mv_grouped_by_lv
        self.__config = config
        self.__correction = correction
        self.__index = data.index

    def iterate(self, inner_weight_calculator: Scheme) -> float:
        self.__betas = {}
        y_old = self.__y.copy()
        inner_weights = inner_weight_calculator.value.calculate(self.__config.path(), self.__y)
        Z = np.dot(self.__y, inner_weights)
        for i, lv in enumerate(list(self.__config.lvs())):
            for j, mv in enumerate(list(self.__config.mvs(lv))):
                self.__mv_grouped_by_lv[lv][:, j] = \
                    self.__config.scale(mv).value.scale(lv, mv, Z[:, i], self)
            self.__weights[lv], self.__y[:, i] = \
                self.__config.mode(lv).value.outer_weights_nonmetric(self.__mv_grouped_by_lv,
                                                                     self.__mv_grouped_by_lv_missing, Z[:, i], lv,
                                                                     self.__correction)
        return np.power(np.abs(y_old) - np.abs(self.__y), 2).sum()

    def calculate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        lvs = self.__config.lvs()
        weights = pd.DataFrame(0, index=self.__mvs, columns=lvs)
        data_new = pd.DataFrame(0, index=self.__index, columns=self.__mvs)
        for lv in self.__config.lvs():
            mvs = self.__config.mvs(lv)
            weights.loc[mvs, [lv]] = self.__weights[lv]
            data_new.loc[:, mvs] = self.__mv_grouped_by_lv[lv]
        weight_factors = 1 / (data_new.dot(weights).std(axis=0, skipna=True) / self.__correction)
        wf_diag = pd.DataFrame(np.diag(weight_factors), index=lvs, columns=lvs)
        weights = weights.dot(wf_diag).sum(axis=1).to_frame(name="weight")
        return data_new, pd.DataFrame(self.__y, index=self.__index, columns=lvs), weights

    def get_Z_for_mode_b(self, lv, mv, z_by_lv):
        mv_index = self.__config.mv_index(lv, mv)
        if self.__config.mode(lv) != Mode.B or len(self.__config.mvs(lv)) == 1:
            return z_by_lv
        if lv not in self.__betas:
            exogenous = sm.add_constant(self.__mv_grouped_by_lv[lv])
            regression = sm.OLS(z_by_lv, exogenous).fit()
            self.__betas[lv] = regression.params[1:]
        correction = 1 / self.__betas[lv][mv_index]
        return correction * (z_by_lv - np.delete(self.__mv_grouped_by_lv[lv], mv_index, axis=1).dot(
            np.delete(self.__betas[lv], mv_index)))

    def correction(self) -> float:
        return self.__correction

    def mv_grouped_by_lv(self, lv: str, mv: str):
        return self.__mv_grouped_by_lv_initial[lv][:, self.__config.mv_index(lv, mv)]

    def dummies(self, mv: str) -> pd.DataFrame:
        return self.__config.dummies(mv)


class WeightsCalculatorFactory:
    """Internal class that is used to calculate weights and scores from the data using the model."""
    def __init__(self, config: c.Config, iterations: int, tolerance: float, correction: float, scheme: Scheme):
        self.__iterations = iterations
        self.__tolerance = tolerance
        self.__config = config
        self.__correction = correction
        self.__scheme = scheme

    def calculate(self, data: pd.DataFrame):
        """Internal method that performs the calculation to estimate weights and scores."""
        if self.__config.metric():
            calculator = _MetricWeights(data, self.__config, self.__correction)
        else:
            calculator = _NonmetricWeights(data, self.__config, self.__correction)

        iteration = 0
        while True:
            iteration += 1
            convergence = calculator.iterate(self.__scheme)
            if (convergence < self.__tolerance) or (iteration > self.__iterations):
                break
        if iteration > self.__iterations:
            raise Exception("Could not converge after " + str(iteration) + " iterations")
        return calculator.calculate()
