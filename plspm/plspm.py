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

import plspm.inner_summary as pis, plspm.config as c
import pandas as pd, numpy as np, plspm.weights as w, plspm.outer_model as om, plspm.inner_model as im
from plspm.scheme import Scheme
from plspm.unidimensionality import Unidimensionality
from plspm.bootstrap import Bootstrap


class Plspm:

    def __init__(self, data: pd.DataFrame, config: c.Config, scheme: Scheme = Scheme.CENTROID,
                 iterations: int = 100, tolerance: float = 0.000001, bootstrap: bool = False,
                 bootstrap_iterations: int = 100):
        if iterations < 100:
            iterations = 100
        assert tolerance > 0
        assert scheme in Scheme
        if bootstrap_iterations < 10:
            bootstrap_iterations = 100

        data_untreated = config.filter(data)
        treated_data = config.treat(data_untreated)
        correction = np.sqrt(treated_data.shape[0] / (treated_data.shape[0] - 1))

        calculator = w.WeightsCalculatorFactory(config, iterations, tolerance, correction, scheme)
        final_data, scores, weights = calculator.calculate(treated_data)

        self.__inner_model = im.InnerModel(config.path(), scores)
        self.__outer_model = om.OuterModel(final_data, scores, weights, config.odm(), self.__inner_model.r_squared())
        self.__inner_summary = pis.InnerSummary(config, self.__inner_model.r_squared(), self.__outer_model.model())
        self.__unidimensionality = Unidimensionality(config, data_untreated, correction)
        self.__scores = scores
        self.__bootstrap = None
        if bootstrap:
            if (treated_data.shape[0] < 10):
                raise Exception("Bootstrapping could not be performed, at least 10 observations are required.")
            self.__bootstrap = Bootstrap(config, data_untreated, self.__inner_model, self.__outer_model, calculator,
                                         bootstrap_iterations)

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

    def effects(self) -> pd.DataFrame:
        return self.__inner_model.effects()

    def unidimensionality(self):
        return self.__unidimensionality.summary()

    def bootstrap(self):
        if self.__bootstrap == None:
            raise Exception("To perform bootstrap validation, set the parameter bootstrap to True when calling Plspm")
        return self.__bootstrap
