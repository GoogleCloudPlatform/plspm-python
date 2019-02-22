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

import pandas as pd, math


def treat(data: pd.DataFrame, center:bool = True, scale:bool = True, scale_values = None):
    if center:
        data = data.subtract(data.mean())
    if scale:
        if scale_values:
            data = data.divide(scale_values)
        else:
            data = data.divide(data.std())
    return data

def sort_cols(data: pd.DataFrame):
    return data.reindex(sorted(data.columns), axis=1)


def impute(data: pd.DataFrame):
    for column in list(data):
        average = data[column].mean(skipna=True)
        data[column].fillna(average, inplace=True)
        assert math.isclose(data[column].mean(), average, rel_tol=1e-09, abs_tol=0.0)
    return data


def list_to_matrix(data: dict):
    matrix = pd.DataFrame()
    for col in data:
        matrix = pd.concat([matrix, data[col]], axis=1, sort=False)
    return matrix.fillna(0)


def list_to_dummy(data: dict):
    matrix = pd.DataFrame()
    for col in data:
        dummy = pd.DataFrame(1, index=data[col], columns=[col])
        matrix = pd.concat([matrix, dummy], axis=1, sort=False)
    return matrix.fillna(0)


def rank(data: pd.Series):
    new = data.copy()
    unique = pd.Series(data.unique())
    ranked = unique.rank()
    for i in range(data.size):
        new.loc[i] = ranked.loc[unique[unique == data.iloc[i]].index[0]]
    return new.astype(int)
