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

import numpy as np, pandas as pd, plspm.util as util, plspm.config as c

pd.options.mode.chained_assignment = None  # default='warn'


class MetricWeights:
    def __init__(self, data: pd.DataFrame, config: c.Config, correction: float, odm: pd.DataFrame):
        weight_factors = correction / data.dot(odm).std(axis=0)
        wf_diag = pd.DataFrame(np.diag(weight_factors), index=weight_factors.index, columns=weight_factors.index)
        weights = odm.dot(wf_diag)
        self.__w_old = weights.sum(axis=1).to_frame(name="weight")
        self.__data = data
        self.__config = config
        self.__weights = weights
        self.__correction = correction

    def iterate(self, inner_weight_calculator):
        y = self.__data.dot(self.__weights)
        y = y.subtract(y.mean()).divide(y.std()) / self.__correction
        inner_weights = inner_weight_calculator.calculate(self.__config.path(), y)
        Z = y.dot(inner_weights)
        for lv in list(y):
            mvs = self.__config.blocks()[lv]
            weights = self.__config.mode(lv).outer_weights_metric(self.__data, Z, lv, mvs)
            self.__weights.loc[mvs, [lv]] = weights
        w_new = self.__weights.sum(axis=1).to_frame(name="weight")
        convergence = np.power(self.__w_old.abs() - w_new.abs(), 2).sum(axis=1).sum(axis=0)
        self.__w_old = w_new
        return convergence

    def calculate(self):
        weight_factors = 1 / (self.__data.dot(self.__weights).std(axis=0) / self.__correction)
        wf_diag = pd.DataFrame(np.diag(weight_factors), index=weight_factors.index, columns=weight_factors.index)
        weights = self.__weights.dot(wf_diag)
        y = self.__data.dot(weights)
        cor = pd.concat([self.__data, y], axis=1).corr().loc[list(self.__data), list(y)]
        odm = weights.apply(lambda x: x != 0).astype(int)
        w_sign = np.sign(np.sign(cor * odm).sum(axis=0))
        if -1 in w_sign.tolist():
            w_sign = [-1 if 0 else x for x in w_sign]
            y = y.dot(np.diag(w_sign), index=weights.index, columns=weights.columns)
        return self.__data, y, weights.sum(axis=1).to_frame(name="weight")


class NonmetricWeights:
    def __init__(self, data: pd.DataFrame, config: c.Config, correction: float):
        mv_grouped_by_lv = {}
        y = pd.DataFrame(data=0, index=data.index, columns=config.blocks().keys())
        for lv in config.blocks():
            mv_grouped_by_lv[lv] = data.filter(config.blocks()[lv])
            sizes = mv_grouped_by_lv[lv].shape[1]
            weight = [1 / np.sqrt(sizes)] * sizes
            y.loc[:, [lv]] = mv_grouped_by_lv[lv].dot(weight)
        self.__weights = {}
        self.__y = y
        self.__mv_grouped_by_lv = mv_grouped_by_lv
        self.__config = config
        self.__correction = correction
        self.__data = data

    def iterate(self, inner_weight_calculator):
        y_old = self.__y.copy()
        inner_weights = inner_weight_calculator.calculate(self.__config.path(), self.__y)
        Z = self.__y.dot(inner_weights)
        for lv in list(self.__y):
            # If not mode B and there is more than one MV in our LV, we're going to scale.
            # This loop is a numerical scaling (get_num_scale in R plspm, get_weights_nonmetric line 162)
            for mv in list(self.__mv_grouped_by_lv[lv]):
                mv_values = self.__mv_grouped_by_lv[lv].loc[:, [mv]]
                self.__mv_grouped_by_lv[lv].loc[:, [mv]] = mv_values * self.__correction / mv_values.std()
                self.__weights[lv], self.__y.loc[:, [lv]] = self.__config.mode(lv).outer_weights_nonmetric(
                    self.__mv_grouped_by_lv, Z, lv, self.__correction)
        return np.power(y_old.abs() - self.__y.abs(), 2).sum(axis=1).sum(axis=0)

    def calculate(self):
        weights = util.list_to_matrix(self.__weights)
        weight_factors = 1 / (self.__data.dot(weights).std(axis=0, skipna=True) / self.__correction)
        wf_diag = pd.DataFrame(np.diag(weight_factors), index=weights.columns, columns=weights.columns)
        weights = weights.dot(wf_diag).sum(axis=1).to_frame(name="weight")
        return util.list_to_matrix(self.__mv_grouped_by_lv), self.__y, weights
