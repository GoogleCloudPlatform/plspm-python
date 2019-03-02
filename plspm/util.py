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

import pandas as pd, math, numpy as np


def treat(data: pd.DataFrame, center: bool = True, scale: bool = True, scale_values=None) -> pd.DataFrame:
    if center:
        data = data.subtract(data.mean())
    if scale:
        if scale_values:
            data = data.divide(scale_values)
        else:
            data = data.divide(data.std())
    return data


def treat_numpy(data: np.ndarray) -> np.ndarray:
    data = data - np.mean(data)
    return data / np.std(data, ddof=1)


def sort_cols(data: pd.DataFrame) -> pd.DataFrame:
    return data.reindex(sorted(data.columns), axis=1)


def impute(data: pd.DataFrame) -> pd.DataFrame:
    for column in list(data):
        average = data[column].mean(skipna=True)
        data[column].fillna(average, inplace=True)
        assert math.isclose(data[column].mean(), average, rel_tol=1e-09, abs_tol=0.0)
    return data


def list_to_dummy(data: dict) -> pd.DataFrame:
    matrix = pd.DataFrame()
    for col in data:
        dummy = pd.DataFrame(1, index=data[col], columns=[col])
        matrix = pd.concat([matrix, dummy], axis=1, sort=False)
    return matrix.fillna(0)


def rank(data: pd.Series) -> pd.Series:
    unique = pd.Series(data.unique())
    ranked = unique.rank()
    lookup = pd.concat([unique, ranked], axis=1)
    lookup_series = pd.Series(lookup.iloc[:, 1].values, index=lookup.iloc[:, 0])
    return data.replace(lookup_series.to_dict()).astype(int)


def dummy(data: pd.Series) -> pd.DataFrame:
    unique = data.unique().size
    dummy = pd.DataFrame(0, data.index, range(1, unique + 1))
    for i in range(unique):
        dummy.loc[data[data == i + 1].index, i + 1] = 1
    return dummy


def groupby_mean(data: np.ndarray) -> np.ndarray:
    values = {}
    reduced = 0
    for i in range(data.shape[1]):
        index = data[0, i]
        if not index in values:
            values[index] = []
            reduced += 1
        values[index].append(data[1, i])
    means = np.zeros((2, reduced), dtype=np.float64)
    for i, index in enumerate(sorted(values.keys())):
        means[0, i] = index
        means[1, i] = np.mean(values[index])
    return means
