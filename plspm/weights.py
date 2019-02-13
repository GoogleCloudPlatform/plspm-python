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
pd.options.mode.chained_assignment = None  # default='warn'

class Weights:
    """Calculate outer weights for partial least squares using Lohmoller's algorithm"""

    def __init__(self, data, blocks):
        self.__inner_model = None
        self.__outer_model = None
        self.__mv_grouped_by_lv = {}
        self.__treated_data = util.treat(data)
        self.__y = pd.DataFrame(data=0, index=data.index, columns=blocks.keys())
        self.__odm = util.list_to_dummy(blocks)

        for lv in blocks:
            self.__mv_grouped_by_lv[lv] = self.__treated_data.filter(blocks[lv])
            sizes = self.__mv_grouped_by_lv[lv].shape[1]
            weight = [1 / np.sqrt(sizes)] * sizes
            self.__y.loc[:, [lv]] = self.__mv_grouped_by_lv[lv].dot(weight)

    # E is inner_weights
    # QQ is mv_grouped_by_lv
    # W is weights
    # Y is y matrix
    def calculate(self, tolerance, max_iterations, inner_weight_calculator, path):
        self.__path = path
        self.__correction = np.sqrt(self.__y.shape[0] / (self.__y.shape[0] - 1))
        iteration = 0
        weights = {}
        y_old = None
        while True:
            iteration += 1
            y_old = self.__y.copy()
            inner_weights = inner_weight_calculator.calculate(path, self.__y)
            Z = self.__y.dot(inner_weights)
            for lv in list(self.__y):
                # If not mode B and there is more than one MV in our LV, we're going to scale.
                # This loop is a numerical scaling (get_num_scale in R plspm, get_weights_nonmetric line 162)
                for mv in list(self.__mv_grouped_by_lv[lv]):
                    mv_values = self.__mv_grouped_by_lv[lv].loc[:, [mv]]
                    self.__mv_grouped_by_lv[lv].loc[:, [mv]] = mv_values * self.__correction / mv_values.std()

                # This is the Mode A algorithm for updating outer weights (get_weights_nonmetric lines 190-192)
                weights[lv] = (self.__mv_grouped_by_lv[lv].transpose().dot(Z.loc[:, [lv]])) / np.power(Z.loc[:, [lv]], 2).sum()
                self.__y.loc[:, [lv]] = self.__mv_grouped_by_lv[lv].dot(weights[lv])
                self.__y.loc[:, [lv]] = self.__y.loc[:, [lv]] * self.__correction / self.__y.loc[:, [lv]].std()
            convergence = np.power(y_old.abs() - self.__y.abs(), 2).sum(axis=1).sum(axis=0)
            if (convergence < tolerance) or (iteration > max_iterations):
                break
        if iteration > max_iterations:
            raise Exception("500 Could not converge after " + str(iteration) + " iterations")
        self.__weights = weights

    def scores(self):
        return self.__y

    def mv_grouped_by_lv(self):
        return self.__mv_grouped_by_lv

    def inner_model(self):
        if (self.__inner_model == None):
            self.__inner_model = im.InnerModel(self.__path, self.__y)
        return self.__inner_model

    def outer_model(self):
        if (self.__outer_model == None):
            self.__outer_model = om.OuterModel(self.__y, self.__weights, self.__mv_grouped_by_lv, self.__odm, self.__correction, self.inner_model().r_squared())
        return self.__outer_model
