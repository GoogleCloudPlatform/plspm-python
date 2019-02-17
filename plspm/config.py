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

import pandas as pd, numpy as np, numpy.testing as npt

class MV:
    def __init__(self, name: str):
        self.__name = name

    def name(self):
        return self.__name


class Config:

    def __init__(self, path: pd.DataFrame):
        self.__modes = {}
        self.__blocks = {}
        if not isinstance(path, pd.DataFrame):
            raise TypeError("path argument must be a Pandas DataFrame")
        path_shape = path.shape
        if path_shape[0] != path_shape[1]:
            raise ValueError("path argument must be a square matrix")
        try:
            npt.assert_array_equal(path, np.tril(path))
        except:
            raise ValueError("path argument must be a lower triangular matrix")
        if not path.isin([0, 1]).all(axis=None):
            raise ValueError("path matrix elements may only be in [0, 1]")
        try:
            npt.assert_array_equal(path.columns.values, path.index.values)
        except:
            raise ValueError("path matrix must have matching row and column index names")
        self.__path = path

    def path(self):
        return self.__path

    def blocks(self):
        return self.__blocks

    def mode(self, lv: str):
        return self.__modes[lv]

    def add_lv(self, name: str, mode, *mvs: MV):
        if name not in self.__path:
            raise ValueError("Path matrix does not contain reference to latent variable " + name)
        self.__modes[name] = mode
        self.__blocks[name] = []
        for mv in mvs:
            self.__blocks[name].append(mv.name())
