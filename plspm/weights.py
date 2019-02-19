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

import numpy as np, pandas as pd, plspm.util as util, plspm.outer_model as om, plspm.inner_model as im
from plspm.config import Config

pd.options.mode.chained_assignment = None  # default='warn'


class Weights:
    """Calculate outer weights for partial least squares using Lohmoller's algorithm"""

    def __init__(self, data: pd.DataFrame, config: Config):
        self.__config = config
        self.__odm = util.list_to_dummy(config.blocks())
        self.__inner_model = None
        self.__outer_model = None
        self.__data = config.filter(data)

    def calculate_metric(self, tolerance: float, max_iterations: int, inner_weight_calculator):
        blocks = self.__config.blocks()
        correction = np.sqrt((self.__data.shape[0] - 1) / self.__data.shape[0])
        weight_factors = 1 / (self.__data.dot(self.__odm).std(axis=0) * correction)
        weights = self.__odm.dot(
            pd.DataFrame(np.diag(weight_factors), index=weight_factors.index, columns=weight_factors.index))
        w_old = weights.sum(axis=1).to_frame(name="weight")
        iteration = 0
        while True:
            iteration += 1
            y = self.__data.dot(weights)
            y = y.subtract(y.mean()).divide(y.std()) * correction
            inner_weights = inner_weight_calculator.calculate(self.__config.path(), y)
            Z = y.dot(inner_weights)
            for lv in list(y):
                # TODO: Mode A only!
                thingy = (1 / self.__data.shape[0]) * Z.loc[:, [lv]].T.dot(self.__data.loc[:, blocks[lv]])
                weights.loc[blocks[lv], [lv]] = thingy.T
            w_new = weights.sum(axis=1).to_frame(name="weight")
            convergence = np.power(w_old.abs() - w_new.abs(), 2).sum(axis=1).sum(axis=0)
            if (convergence < tolerance) or (iteration > max_iterations):
                break
            w_old = w_new
        if iteration > max_iterations:
            raise Exception("500 Could not converge after " + str(iteration) + " iterations")
        weight_factors = 1 / (self.__data.dot(weights).std(axis=0) * correction)
        weights = weights.dot(
            pd.DataFrame(np.diag(weight_factors), index=weight_factors.index, columns=weight_factors.index))
        self.__weights = weights.sum(axis=1).to_frame(name="weight")
        y = self.__data.dot(weights)
        cor = pd.concat([self.__data, y], axis=1).corr().loc[list(self.__data), list(y)]
        odm = weights.apply(lambda x: x != 0).astype(int)
        w_sign = np.sign(np.sign(cor * odm).sum(axis=0))
        if -1 in w_sign.tolist():
            w_sign = [-1 if 0 else x for x in w_sign]
            y = y.dot(np.diag(w_sign), index=weights.index, columns=weights.columns)
        self.__scores = y

    # E is inner_weights
    # QQ is mv_grouped_by_lv
    # W is weights
    # Y is y matrix (outer estimates)
    # Z is inner estimate of LV (Y.dot(inner_weights))
    def calculate(self, tolerance: float, max_iterations: int, inner_weight_calculator):
        data = util.treat(self.__data)
        rows = data.shape[0]
        rank = np.sqrt((rows - 1.0) / rows)
        data = data.apply(lambda x: x / rank)
        blocks = self.__config.blocks()

        mv_grouped_by_lv = {}
        y = pd.DataFrame(data=0, index=data.index, columns=blocks.keys())
        for lv in blocks:
            mv_grouped_by_lv[lv] = data.filter(blocks[lv])
            sizes = mv_grouped_by_lv[lv].shape[1]
            weight = [1 / np.sqrt(sizes)] * sizes
            y.loc[:, [lv]] = mv_grouped_by_lv[lv].dot(weight)
        correction = np.sqrt(y.shape[0] / (y.shape[0] - 1))
        weights = {}
        iteration = 0
        while True:
            iteration += 1
            y_old = y.copy()
            inner_weights = inner_weight_calculator.calculate(self.__config.path(), y)
            Z = y.dot(inner_weights)
            for lv in list(y):
                # If not mode B and there is more than one MV in our LV, we're going to scale.
                # This loop is a numerical scaling (get_num_scale in R plspm, get_weights_nonmetric line 162)
                for mv in list(mv_grouped_by_lv[lv]):
                    mv_values = mv_grouped_by_lv[lv].loc[:, [mv]]
                    mv_grouped_by_lv[lv].loc[:, [mv]] = mv_values * correction / mv_values.std()

                weights[lv], y.loc[:, [lv]] = self.__config.mode(lv).update_outer_weights(mv_grouped_by_lv, Z, lv,
                                                                                          correction)
            convergence = np.power(y_old.abs() - y.abs(), 2).sum(axis=1).sum(axis=0)
            if (convergence < tolerance) or (iteration > max_iterations):
                break
        if iteration > max_iterations:
            raise Exception("500 Could not converge after " + str(iteration) + " iterations")
        self.__scores = y
        self.__weights = weights
        self.__correction = correction
        self.__data = util.list_to_matrix(mv_grouped_by_lv)

    def scores(self):
        return self.__scores

    def inner_model(self):
        if (self.__inner_model == None):
            self.__inner_model = im.InnerModel(self.__config.path(), self.__scores)
        return self.__inner_model

    def outer_model(self):
        if (self.__outer_model == None):
            self.__outer_model = om.OuterModel(self.__data, self.__scores, self.__weights, self.__odm,
                                               self.__correction, self.inner_model().r_squared())
        return self.__outer_model
