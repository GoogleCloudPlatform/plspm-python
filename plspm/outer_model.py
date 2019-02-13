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

import plspm.util as util, pandas as pd, numpy as np

class OuterModel:

    def __init__(self, y, weights, mv_grouped_by_lv, odm, correction, r_squared):
        weights_as_matrix = util.list_to_matrix(weights)
        quantified_mvs = util.list_to_matrix(mv_grouped_by_lv)
        weight_factors = 1 / (quantified_mvs.dot(weights_as_matrix).std(axis=0, skipna=True) / correction)
        weight = weights_as_matrix.dot(pd.DataFrame(np.diag(weight_factors), index=weight_factors.index, columns=weight_factors.index)).sum(axis=1).to_frame(name="weight")
        self.__crossloadings = y.apply(lambda s: quantified_mvs.corrwith(s))
        loading = (self.__crossloadings * odm).sum(axis=1).to_frame(name="loading")
        communality = loading.apply(lambda s: pow(s, 2))
        communality.columns = ["communality"]
        r_squared_aux = odm.dot(pd.DataFrame(np.diag(r_squared), index=r_squared.index, columns=r_squared.index)).sum(axis=1).to_frame(name="communality")
        redundancy = communality * r_squared_aux
        redundancy.columns = ["redundancy"]
        self.__outer_model = pd.concat([weight, loading, communality, redundancy], axis=1, sort=True)

    def model(self):
        return self.__outer_model

    def crossloadings(self):
        return self.__crossloadings
