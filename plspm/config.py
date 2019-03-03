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

import pandas as pd, numpy as np, numpy.testing as npt, plspm.util as util
from plspm.mode import Mode
from plspm.scale import Scale


class MV:
    def __init__(self, name: str, scale: Scale = None):
        self.__scale = scale
        self.__name = name

    def name(self):
        return self.__name

    def scale(self):
        return self.__scale


class Config:
    def __init__(self, path: pd.DataFrame, scaled: bool = True, default_scale: Scale = None):
        self.__modes = {}
        self.__mvs = {}
        self.__dummies = {}
        self.__mv_scales = {}
        self.__scaled = scaled
        self.__metric = True
        self.__default_scale = default_scale
        if not isinstance(path, pd.DataFrame):
            raise TypeError("Path argument must be a Pandas DataFrame")
        path_shape = path.shape
        if path_shape[0] != path_shape[1]:
            raise ValueError("Path argument must be a square matrix")
        try:
            npt.assert_array_equal(path, np.tril(path))
        except:
            raise ValueError("Path argument must be a lower triangular matrix")
        if not path.isin([0, 1]).all(axis=None):
            raise ValueError("Path matrix element values may only be in [0, 1]")
        try:
            npt.assert_array_equal(path.columns.values, path.index.values)
        except:
            raise ValueError("Path matrix must have matching row and column index names")
        self.__path = path

    def path(self):
        return self.__path

    def odm(self):
        return util.list_to_dummy(self.__mvs)

    def mv_index(self, lv, mv):
        return self.__mvs[lv].index(mv)

    def mvs(self, lv):
        return self.__mvs[lv]

    def lvs(self):
        return self.__mvs.keys()

    def mode(self, lv: str):
        return self.__modes[lv]

    def metric(self):
        return self.__metric

    def scaled(self):
        return self.__scaled

    def scale(self, mv: str):
        return self.__mv_scales[mv]

    def dummies(self, mv: str):
        return self.__dummies[mv]

    def add_lv(self, name: str, mode: Mode, *mvs: MV):
        assert mode in Mode
        if name not in self.__path:
            raise ValueError("Path matrix does not contain reference to latent variable " + name)
        self.__modes[name] = mode
        self.__mvs[name] = []
        for mv in mvs:
            self.__mvs[name].append(mv.name())
            scale = self.__default_scale if mv.scale() is None else mv.scale()
            self.__mv_scales[mv.name()] = scale
            if scale is not None:
                self.__metric = False

    def add_lv_with_columns_named(self, lv_name: str, mode: Mode, data: pd.DataFrame, col_name_starts_with: str,
                                  default_scale: Scale = None):
        names = filter(lambda x: x.startswith(col_name_starts_with), list(data))
        mvs = list(map(lambda mv: MV(mv, default_scale), names))
        self.add_lv(lv_name, mode, *mvs)

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        if not set(self.__mv_scales.keys()).issubset(set(data)):
            raise ValueError(
                "The following manifest variables you configured are not present in the data set: " + ", ".join(
                    set(self.__mv_scales.keys()).difference(set(data))))
        data = data[list(self.__mv_scales.keys())]
        if False in data.apply(lambda x: np.issubdtype(x.dtype, np.number)).values:
            raise ValueError(
                "Data must only contain numeric values. Please convert any categorical data into numerical values.")
        return data

    def treat(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.__metric:
            if self.__scaled:
                scale_values = data.stack().std() * np.sqrt((data.shape[0] - 1) / data.shape[0])
                return util.treat(data, scale_values=scale_values)
            else:
                return util.treat(data, scale=False)
        else:
            if None in self.__mv_scales.values():
                raise TypeError(
                    "If you supply a scale for any MV, you must either supply a scale for all of them or specify a default scale.")
            if set(self.__mv_scales.values()) == {Scale.RAW}:
                self.__scaled = False
            if set(self.__mv_scales.values()) == {Scale.RAW, Scale.NUM}:
                self.__scaled = True
                self.__mv_scales = dict.fromkeys(self.__mv_scales, Scale.NUM)
            data = util.treat(data) / np.sqrt((data.shape[0] - 1) / data.shape[0])
            for mv in self.__mv_scales:
                if self.__mv_scales[mv] in [Scale.ORD, Scale.NOM]:
                    data.loc[:, mv] = util.rank(data.loc[:, mv])
                    self.__dummies[mv] = util.dummy(data.loc[:, mv]).values
            return data
