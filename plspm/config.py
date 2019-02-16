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

class Config:

    def __init__(self, path, lv_config):
        if not isinstance(path, pd.DataFrame):
            raise TypeError("path argument must be a Pandas DataFrame")
        if not isinstance(lv_config, dict):
            raise TypeError("lv_config argument must be a dictionary")
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
        lv_config_keys = pd.Series(list(lv_config.keys())).sort_values()
        path_column_titles = pd.Series(path.columns.values).sort_values()
        try:
            npt.assert_array_equal(lv_config_keys, path_column_titles)
        except:
            raise ValueError("path matrix and lv_config must have matching keys")
        self.__path = path
        self.__lv_config = lv_config
