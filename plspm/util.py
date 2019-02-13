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

import numpy as np, pandas as pd, math

def treat(data):
    """Centers, scales and ranks a matrix"""
    rows = data.shape[0]
    rank = np.sqrt((rows - 1.0) / rows)
    return data.subtract(data.mean()).divide(data.std()).apply(lambda x : x / rank)

def sort_cols(data):
    return data.reindex(sorted(data.columns), axis=1)

def impute(data):
    for column in list(data):
        average = data[column].mean(skipna=True)
        data[column].fillna(average, inplace=True)
        assert math.isclose(data[column].mean(), average, rel_tol=1e-09, abs_tol=0.0)
    return data

def list_to_matrix(data):
    matrix = pd.DataFrame()
    for col in data:
        matrix = pd.concat([matrix, data[col]], axis=1, sort=False)
    return matrix.fillna(0)

def list_to_dummy(data):
    matrix = pd.DataFrame()
    for col in data:
        dummy = pd.DataFrame(1, index=data[col], columns=[col])
        matrix = pd.concat([matrix, dummy], axis=1, sort=False)
    return matrix.fillna(0)
