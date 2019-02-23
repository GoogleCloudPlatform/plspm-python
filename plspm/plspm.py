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

import plspm.inner_summary as pis, plspm.config as c, plspm.util as util
import pandas as pd, numpy as np, plspm.weights as w, plspm.outer_model as om, plspm.inner_model as im
from plspm.scheme import Scheme


class Plspm:

    def __init__(self, input_data: pd.DataFrame, config: c.Config, scheme: Scheme = Scheme.CENTROID,
                 iterations: int = 100, tolerance: float = 0.000001):
        if iterations < 100:
            iterations = 100
        assert tolerance > 0
        assert scheme in Scheme

        data = config.filter(input_data)
        correction = np.sqrt(data.shape[0] / (data.shape[0] - 1))
        odm = util.list_to_dummy(config.blocks())

        if config.metric():
            calculator = w.MetricWeights(data, config, correction, odm)
        else:
            calculator = w.NonmetricWeights(data, config, correction)

        iteration = 0
        while True:
            iteration += 1
            convergence = calculator.iterate(scheme)
            if (convergence < tolerance) or (iteration > iterations):
                break
        if iteration > iterations:
            raise Exception("Could not converge after " + str(iteration) + " iterations")
        data, scores, weights = calculator.calculate()

        self.__inner_model = im.InnerModel(config.path(), scores)
        self.__outer_model = om.OuterModel(data, scores, weights, odm, self.__inner_model.r_squared())
        self.__inner_summary = pis.InnerSummary(config, self.__inner_model.r_squared(), self.__outer_model.model())
        self.__scores = scores

    def scores(self):
        return self.__scores

    def outer_model(self) -> pd.DataFrame:
        return self.__outer_model.model()

    def inner_model(self) -> dict:
        return self.__inner_model.inner_model()

    def path_coefficients(self) -> pd.DataFrame:
        return self.__inner_model.path_coefficients()

    def crossloadings(self) -> pd.DataFrame:
        return self.__outer_model.crossloadings()

    def inner_summary(self) -> pd.DataFrame:
        return self.__inner_summary.summary()

    def goodness_of_fit(self) -> float:
        return self.__inner_summary.goodness_of_fit()
