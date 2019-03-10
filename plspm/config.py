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
    """Specify a manifest variable to use in the model.

    You must create an instance of this class for every manifest variable when using :meth:`add_lv` to add latent variables to the model.
    """
    def __init__(self, name: str, scale: Scale = None):
        """Specify a manifest variable to use in the model.

        Args:
            name: The name of the manifest variable (must match the column name corresponding to this manifest variable in the dataset).
            scale: If using nonmetric data, the measurement type of the variable (otherwise ``None``). Takes a value from the enum :class:`~plspm.scale.Scale`. Note that if you specify a scale for one variable, you must specify a scale for all variables (or specify a default scale to use in the constructor).
        """
        self.__scale = scale
        self.__name = name

    def name(self):
        """Internal method that returns the name of the manifest variable"""
        return self.__name

    def scale(self):
        """Internal method that returns the scale of the manifest variable (if specified)"""
        return self.__scale


class Config:
    """Specify the model you want to calculate with :class:`~.plspm.Plspm`

    Create an instance of this class in order to specify the model you want to use :class:`~.plspm.Plspm` to calculate.
    """
    def __init__(self, path: pd.DataFrame, scaled: bool = True, default_scale: Scale = None):
        """Specify the model you want to calculate with :mod:~.plspm.plspm:

        Once you have created an instance of this class, add the relevant latent and manifest variables with :meth:`add_lv` or :meth:`add_lv_add_lv_with_columns_named`

        Args:
            path: A square lower triangular matrix which specifies the paths between the latent variables (the inner model). The index and columns of this matrix must be the same, and must consist of the names of the latent variables. Cells should contain 1 if the variable in the column affects the variable in the row, 0 otherwise.
            scaled: Whether manifest variables should be standardized. When ``True``, data is scaled to standardized values (mean 0 and variance 1). Only used when ``default_scale`` is set to ``None``.
            default_scale: If your data is nonmetric, specify a default measurement type. Takes a value from the enum :class:`~plspm.scale.Scale`
        """
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
        """Internal method that returns the matrix of paths provided in the constructor."""
        return self.__path

    def odm(self):
        """Internal method that returns the outer design matrix showing which manifest variables belong to the latent variables in the model."""
        return util.list_to_dummy(self.__mvs)

    def mv_index(self, lv, mv):
        """Internal method that returns the index of a manifest variable for a given latent variable."""
        return self.__mvs[lv].index(mv)

    def mvs(self, lv):
        """Internal method that returns the manifest variables belonging to a given latent variable."""
        return self.__mvs[lv]

    def lvs(self):
        """Internal method that returns a list of the latent variables in the model."""
        return self.__mvs.keys()

    def mode(self, lv: str):
        """Internal method that returns the mode of a given latent variable."""
        return self.__modes[lv]

    def metric(self):
        """Internal method that returns whether we are using metric or nonmetric data."""
        return self.__metric

    def scaled(self):
        """Internal method that returns whether the user asked for the data to be scaled."""
        return self.__scaled

    def scale(self, mv: str):
        """Internal method that returns the scale for a given manifest variable."""
        return self.__mv_scales[mv]

    def dummies(self, mv: str):
        """Internal method that returns a dummy matrix which is used for handling ordinal or nominal data"""
        return self.__dummies[mv]

    def add_lv(self, lv_name: str, mode: Mode, *mvs: MV):
        """Add a latent variable and associated manifest variables to the model.

        Args:
            lv_name: The name of the latent variable to add. Must match the name used in the columns / index of the Path matrix passed into the constructor.
            mode: Whether the latent variable is reflective (mode A) or formative (mode B) with respect to its manifest variables. Takes a value from the enum :class:`~plspm.mode.Mode`
            *mvs: A list of manifest variables that make up the latent variable. These must be instances of :class:`MV`
        """
        assert mode in Mode
        if lv_name not in self.__path:
            raise ValueError("Path matrix does not contain reference to latent variable " + lv_name)
        self.__modes[lv_name] = mode
        self.__mvs[lv_name] = []
        for mv in mvs:
            self.__mvs[lv_name].append(mv.name())
            scale = self.__default_scale if mv.scale() is None else mv.scale()
            self.__mv_scales[mv.name()] = scale
            if scale is not None:
                self.__metric = False

    def add_lv_with_columns_named(self, lv_name: str, mode: Mode, data: pd.DataFrame, col_name_starts_with: str,
                                  default_scale: Scale = None):
        """Add a latent variable and associated manifest variables to the model.

        This is a convenience method that can be used if the names of the columns in the dataset corresponding to the manifest variables all share a common prefix, and are all the same :class:`.Scale` (if nonmetric).

        Args:
            lv_name: The name of the latent variable to add. Must match the name used in the columns / index of the Path matrix passed into the constructor.
            mode: Whether the latent variable is reflective (mode A) or formative (mode B) with respect to its manifest variables. Takes a value from the enum :class:`~plspm.mode.Mode`
            data: The dataset that will be passed into :class:`.Plspm`
            col_name_starts_with: The prefix for the column names in the dataset corresponding to the manifest variables. For example, if the columns are named ``var1``, ``var2``, ``var3``, use ``var``
            default_scale: If the data is nonmetric, the measurement type of the manifest variables. Note that you can only use this method if all the manifest variables are the same type of measurement, otherwise use :meth:`add_lv`. Takes a value from the enum :class:`~plspm.scale.Scale`. ``None`` (the default) if the data is metric.
        """
        names = filter(lambda x: x.startswith(col_name_starts_with), list(data))
        mvs = list(map(lambda mv: MV(mv, default_scale), names))
        self.add_lv(lv_name, mode, *mvs)

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Internal method that removes columns from the dataset that are not specified in the model.

        Args:
            data: The dataset to filter

        Returns:
            The dataset with columns not specified in the columns removed.

        Raises:
            ValueError: if the dataset is missing any columns with names that were specified as manifest variables in the model, or if there are any non-numeric values in the dataset.
        """
        if not set(self.__mv_scales.keys()).issubset(set(data)):
            raise ValueError(
                "The following manifest variables you configured are not present in the data set: " + ", ".join(
                    set(self.__mv_scales.keys()).difference(set(data))))
        data = data[list(self.__mv_scales.keys())]
        if False in data.apply(lambda x: np.issubdtype(x.dtype, np.number)).values:
            raise ValueError(
                "Data must only contain numeric values. Please convert any categorical data into numerical values.")
        self.__missing = data.isnull().values.any()
        return data

    def treat(self, data: pd.DataFrame) -> pd.DataFrame:
        """Internal method that treats the data (including scaling, normalizing, standardizing and rankifying, where appropriate)

        Args:
            data: The dataset to treat.

        Returns:
            The treated dataset.

        Raises:
            TypeError: if you have specified a scale for some but not all manifest variables. Specifying a scale for any MV tells Plspm that you are using nonmetric data, which means you must specify a scale for all MVs (or specify a default scale in the constructor).
        """
        if self.__metric:
            metric_data = util.impute(data) if self.__missing else data
            if self.__scaled:
                scale_values = metric_data.stack().std() * np.sqrt((metric_data.shape[0] - 1) / metric_data.shape[0])
                return util.treat(metric_data, scale_values=scale_values)
            else:
                return util.treat(metric_data, scale=False)
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
