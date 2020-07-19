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

import pandas as pd, numpy as np, numpy.testing as npt, plspm.util as util, itertools as it, collections as c
from plspm.util import TopoSort
from plspm.mode import Mode
from plspm.scale import Scale


class Structure:
    """Specify relationships betweeen constructs


    Use this class to specify the relationships between constructs. It will generate a path matrix suitable for using in :class:`~.plspm.config.Config`.
    """
    def __init__(self, path: pd.DataFrame = None):
        self.__toposort = TopoSort()
        if path is not None:
            paths = [(path.columns[y], path.index[x]) for x, y in zip(*np.where(path.values == 1))]
            for my_path in paths:
                self.add_path([my_path[0]], [my_path[1]])

    def add_path(self, source: list, target: list):
        """Specify a relationship between two sets of constructs.

        Args:
            source: A list of antecedent constructs
            target: A list of outcome constructs
        """
        if len(source) != 1 and len(target) != 1:
            raise ValueError("Either source or target must be a list containing a single entry")
        if len(source) == 0 or len(target) == 0:
            raise ValueError("Both source and target must contain at least one entry")
        for element in it.product(source, target):
            self.__toposort.append(element[0], element[1])

    def path(self):
        """Get a path matrix for use in :class:`~plspm.Config`.
        """
        index = self.__toposort.order()
        path = pd.DataFrame(np.zeros((len(index), len(index)), int), columns=index, index=index)
        for source, target in self.__toposort.elements():
            path.at[target, source] = 1
        return path

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
        """Specify the model you want to calculate with :class:`~.plspm.Plspm`:

        Once you have created an instance of this class, add the relevant latent and manifest variables with :meth:`add_lv` or :meth:`add_lv_with_columns_named`

        Args:
            path: A matrix specifying the inner model. You can create this using :class:`~plspm.config.Structure`. This square lower triangular matrix specifies the paths between the latent variables (the inner model). The index and columns of this matrix must be the same, and must consist of the names of the latent variables. Cells should contain 1 if the variable in the column affects the variable in the row, 0 otherwise.
            scaled: Whether manifest variables should be standardized. When ``True``, data is scaled to standardized values (mean 0 and variance 1). Only used when ``default_scale`` is set to ``None``.
            default_scale: If your data is nonmetric, specify a default measurement type. Takes a value from the enum :class:`~plspm.scale.Scale`, or ``None`` (the default) if the data is metric and does not require scaling.
        """
        self.__modes = {}
        self.__mvs = {}
        self.__hoc = {}
        self.__dummies = {}
        self.__mv_scales = {}
        self.__scaled = scaled
        self.__metric = True
        self.__default_scale = default_scale
        self.__missing = False
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

    def clone(self):
        my_clone = Config(self.__path, self.__scaled, self.__default_scale)
        my_clone.__modes = self.__modes.copy()
        my_clone.__mvs = self.__mvs.copy()
        my_clone.__hoc = self.__hoc.copy()
        my_clone.__dummies = self.__dummies.copy()
        my_clone.__mv_scales = self.__mv_scales.copy()
        my_clone.__metric = self.__metric
        my_clone.__missing = self.__missing
        return my_clone

    def path(self):
        """Internal method that returns the matrix of paths provided in the constructor."""
        return self.__path

    def odm(self, path: pd.DataFrame):
        """Internal method that returns the outer design matrix showing which manifest variables belong to which latent variables in the model."""
        # Filter out LVs that aren't in path matrix
        mvs = { key: self.__mvs[key] for key in list(path) }
        return util.list_to_dummy(mvs)

    def mv_index(self, lv, mv):
        """Internal method that returns the index of a manifest variable for a given latent variable."""
        return self.__mvs[lv].index(mv)

    def mvs(self, lv):
        """Internal method that returns the manifest variables belonging to a given latent variable."""
        return self.__mvs[lv]

    def hoc(self):
        """Internal method that returns a dictionary with the higher order constructs in the model as the key and the list of constituent LVs as the value."""
        return self.__hoc

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
        hoc_lvs = [item for lvs in self.__hoc.values() for item in lvs]
        if lv_name not in self.__path and lv_name not in hoc_lvs:
            raise ValueError("Latent variable " + lv_name + " is not listed in the outer model paths or higher order constructs.")
        self.__modes[lv_name] = mode
        self.__mvs[lv_name] = []
        for mv in mvs:
            if mv.name() in self.__mv_scales:
                raise ValueError("You can only specify a column once. You can specify a higher order construct with `add_higher_order(...)`")
            if mv.name() in list(self.__path):
                raise ValueError("You cannot specify MVs with the same name as LVs.")
            self.__mvs[lv_name].append(mv.name())
            scale = self.__default_scale if mv.scale() is None else mv.scale()
            self.__mv_scales[mv.name()] = scale
            if scale is not None:
                self.__metric = False

    def remove_lv(self, lv_name:str):
        """Remove a latent variable and associated manifest variables from the model.

        Args:
            lv_name: The name of the latent variable to remove.
        """
        self.__mvs.pop(lv_name)
        self.__modes.pop(lv_name)
        
    def add_higher_order(self, hoc_name: str, mode: Mode, lvs: list):
        """Add a higher order construct to the model.

        Args:
            name: The name of the higher order constuct to add. Must match the name used in the columns / index of the Path matrix passed into the constructor.
            mode: Whether to use correlation weights (mode A) or regression weights (mode B) with respect to the relationship between the second order construct and its component first order constructs.
            method: Estimation method to use
            lvs: A list of the first order constructs that comprise this second order construct
        """
        # TODO: Warn if centroid scheme is used with HOC.
        assert mode in Mode
        if hoc_name not in self.__path:
            raise ValueError("Path matrix does not contain reference to higher order construct " + hoc_name)
        self.__modes[hoc_name] = mode
        self.__hoc[hoc_name] = lvs

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
        if len(mvs) == 0:
            raise ValueError("No columns were found in the data starting with " + col_name_starts_with)
        self.add_lv(lv_name, mode, *mvs)

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Internal method that removes columns from the dataset that are not specified in the model, and rows where all MVs for an LV are unknown.

        Args:
            data: The dataset to filter

        Returns:
            The dataset with columns not specified in the columns removed.

        Raises:
            ValueError: if the dataset is missing any columns with names that were specified as manifest variables in the model, or if there are any non-numeric values in the dataset.
        """
        hoc_lvs = [item for lvs in self.__hoc.values() for item in lvs]
        path_lvs = filter(lambda i: i not in self.__hoc.keys(), list(self.path()) + hoc_lvs)
        if set(self.__mvs.keys()) != set(path_lvs):
            raise ValueError(
                "The Path matrix supplied does not specify the same latent variables as you added when configuring manifest variables." +
                " Path: " + ", ".join(path_lvs) + " LVs: " + ", ".join(set(self.__mvs.keys())))
        if not set(self.__mv_scales.keys()).issubset(set(data)):
            raise ValueError(
                "The following manifest variables you configured are not present in the data set: " + ", ".join(
                    set(self.__mv_scales.keys()).difference(set(data))))
        data = data[list(self.__mv_scales.keys())]
        if False in data.apply(lambda x: np.issubdtype(x.dtype, np.number)).values:
            raise ValueError(
                "Data must only contain numeric values. Please convert any categorical data into numerical values.")
        self.__missing = data.isnull().values.any()
        # Delete any rows which has all MVs for an LV as NaN
        if self.__missing:
            mv_grouped_by_lv = {}
            rows_to_delete = set()
            for i, lv in enumerate(list(self.path())):
                mvs = self.mvs(lv)
                mv_grouped_by_lv[lv] = data.filter(mvs).values.astype(np.float64)
                for j in range(len(data.index)):
                    if np.count_nonzero(~np.isnan(mv_grouped_by_lv[lv][j, :])) == 0:
                        rows_to_delete.add(j)
            data = data.drop(data.index[list(rows_to_delete)])
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
                raise TypeError("If you supply a scale for any MV, you must either supply a scale for all of them or specify a default scale.")
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
