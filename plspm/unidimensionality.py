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

import pandas as pd, numpy as np, plspm.util as util
from sklearn.decomposition import PCA
from plspm.config import Config
from plspm.mode import Mode

class Unidimensionality:
    """Internal class that computes various reliability metrics. Use the method :meth:`~.plspm.Plspm.unidimensionality` defined on :class:`~.plspm.Plspm` to retrieve the results."""
    def __init__(self, config: Config, data: pd.DataFrame, correction: float):
        self.__config = config
        self.__data = data
        self.__correction = correction

    def summary(self):
        """Internal method that performs principal component analysis to compute various reliability metrics."""
        summary = pd.DataFrame(np.NaN, index=self.__config.lvs(),
                               columns=["mode", "mvs", "cronbach_alpha", "dillon_goldstein_rho", "eig_1st", "eig_2nd"])
        for lv in self.__config.lvs():
            mvs = len(self.__config.mvs(lv))
            summary.loc[lv, "mode"] = self.__config.mode(lv).name
            summary.loc[lv, "mvs"] = mvs
            if not self.__data.loc[:,self.__config.mvs(lv)].isnull().values.any():
                mvs_for_lvs = util.treat(self.__data.filter(self.__config.mvs(lv))) * self.__correction
                pca_input = mvs_for_lvs if mvs_for_lvs.shape[0] > mvs_for_lvs.shape[1] else mvs_for_lvs.transpose()
                pca = PCA()
                pca_scores = pca.fit_transform(pca_input)
                pca_std_dev = np.std(pca_scores, axis=0)
                summary.loc[lv, "eig_1st"] = pca_std_dev[0] ** 2
                summary.loc[lv, "eig_2nd"] = pca_std_dev[1] ** 2
                if (self.__config.mode(lv) == Mode.A):
                    ca_numerator = 2 * np.tril(pca_input.corr(), -1).sum()
                    ca_denominator = pca_input.sum(axis=1).var() / self.__correction ** 2
                    ca = (ca_numerator / ca_denominator) * (mvs / (mvs - 1))
                    summary.loc[lv, "cronbach_alpha"] = ca if ca > 0 else 0
                    corr = np.corrcoef(np.column_stack((pca_input.values, pca_scores[:,0])), rowvar=False)[:,-1][:-1]
                    rho_numerator = sum(corr) ** 2
                    rho_denominator = rho_numerator + (mvs - np.sum(np.power(corr, 2)))
                    summary.loc[lv, "dillon_goldstein_rho"] = rho_numerator / rho_denominator
        return summary