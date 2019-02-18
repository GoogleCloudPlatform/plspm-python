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

import plspm.weights as ow, plspm.inner_summary as pis, pandas as pd, plspm.config as c, plspm.scheme as scheme


class Plspm:

    def __init__(self, input_data: pd.DataFrame, config: c.Config, scheme = scheme.CENTROID, iterations: int = 100,
                 tolerance: float = 0.000001):
        if iterations < 100:
            iterations = 100
        assert tolerance > 0

        self.__config = config
        self.__outer_weights = ow.Weights(input_data, config)
        self.__outer_weights.calculate(tolerance, iterations, scheme)

    def scores(self):
        return self.__outer_weights.scores()

    def outer_model(self):
        return self.__outer_weights.outer_model().model()

    def inner_model(self):
        return self.__outer_weights.inner_model().inner_model()

    def path_coefficients(self):
        return self.__outer_weights.inner_model().path_coefficients()

    def crossloadings(self):
        return self.__outer_weights.outer_model().crossloadings()

    def inner_summary(self):
        return pis.InnerSummary(self.__config, self.__outer_weights.inner_model(), self.outer_model()).summary()
