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

import pandas as pd, math, numpy as np, collections as c


def treat(data: pd.DataFrame, center: bool = True, scale: bool = True, scale_values=None) -> pd.DataFrame:
    """Internal function that treats data in Pandas Dataframe format.

    Args:
        data: The data to treat
        center: Whether to center the data
        scale: Whether to scale the data
        scale_values: The scaling to use

    Returns:
        The treated data
    """
    if center:
        data = data.subtract(data.mean())
    if scale:
        if scale_values:
            data = data.divide(scale_values)
        else:
            data = data.divide(data.std())
    return data


def treat_numpy(data: np.ndarray) -> np.ndarray:
    """Internal function that centers and scales data in Numpy format.

    Args:
        data: The data to treat

    Returns:
        The treated data
    """
    data = data - np.nanmean(data)
    return data / np.nanstd(data, axis=0, ddof=1)


def sort_cols(data: pd.DataFrame) -> pd.DataFrame:
    """Internal convenience function to sort data by column."""
    return data.reindex(sorted(data.columns), axis=1)


def impute(data: pd.DataFrame) -> pd.DataFrame:
    """Internal function that imputes missing data using the mean value (only suitable for metric data)."""
    imputed = pd.DataFrame(0, data.index, data.columns)
    for column in list(data):
        average = data[column].mean(skipna=True)
        imputed[column] = data[column].fillna(average)
        assert math.isclose(imputed[column].mean(), average, rel_tol=1e-09, abs_tol=0.0)
    return imputed


def list_to_dummy(data: dict) -> pd.DataFrame:
    """Internal function used to create the outer design matrix."""
    matrix = pd.DataFrame()
    for col in data:
        dummy = pd.DataFrame(1, index=data[col], columns=[col])
        matrix = pd.concat([matrix, dummy], axis=1, sort=False)
    return matrix.fillna(0)


def rank(data: pd.Series) -> pd.Series:
    """Internal function used to rank ordinal and nominal data."""
    unique = pd.Series(data.unique())
    ranked = unique.rank()
    lookup = pd.concat([unique, ranked], axis=1)
    lookup_series = pd.Series(lookup.iloc[:, 1].values, index=lookup.iloc[:, 0])
    return data.replace(lookup_series.to_dict()).astype(float)


def dummy(data: pd.Series) -> pd.DataFrame:
    """Internal function used to create a dummy matrix used to perform calculations with ordinal and nominal data"""
    unique = data.unique().size
    dummy = pd.DataFrame(0, data.index, range(1, unique + 1))
    for i in range(unique):
        dummy.loc[data[data == i + 1].index, i + 1] = 1
    return dummy


def groupby_mean(data: np.ndarray) -> np.ndarray:
    """Internal function which performs the Numpy equivalent of Pandas ``.groupby(...).mean()``"""
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


class TopoSort:
    """Internal fucntion which performs a topological sort using Kahn's algorithm"""

    def __init__(self):
        self.__indegree = c.Counter()
        self.__children = {}
        self.__edges = []

    def append(self, src: str, dest: str):
        self.__edges.append((src, dest))
        self.__indegree[dest] += 1
        self.__indegree[src] += 0
        for vertex in [src, dest]:
            if vertex not in self.__children:
                self.__children[vertex] = []
        self.__children[src].append(dest)

    def order(self):
        ordered = []
        orphaned = c.deque([v for v in self.__indegree if self.__indegree[v] == 0])
        while orphaned:
            vertex = orphaned.pop()
            ordered.append(vertex)
            for child in self.__children[vertex]:
                self.__indegree[child] -= 1
                if self.__indegree[child] == 0:
                    orphaned.append(child)
        for v in self.__indegree:
            if self.__indegree[v] != 0:
                raise ValueError("Structural graph contains cycles.")
        return ordered

    def elements(self):
        return self.__edges
