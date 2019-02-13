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

import pandas as pd, plspm.scheme as scheme, plspm.weights as ow, plspm.inner_summary as pis

class Plspm:

    def __init__(self, input_data, path, blocks, scheme, iterations = 100, tolerance = 0.000001):
        self.__path = path
        self.__blocks = blocks
        self.__outer_weights = ow.Weights(input_data, blocks)
        self.__outer_weights.calculate(tolerance, iterations, scheme, path)

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
        return pis.InnerSummary(self.__path, self.__blocks, self.__outer_weights.inner_model(), self.outer_model()).summary()
