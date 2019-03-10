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

import pandas as pd, numpy as np
from plspm.config import Config
from plspm.mode import Mode


class InnerSummary:
    """Internal class that computes a summary of the inner model.  Use the methods :meth:`~plspm.Plspm.inner_summary` and :meth:`~plspm.Plspm.goodness_of_fit` defined on :class:`~.plspm.Plspm` to retrieve the inner model characteristics."""

    def __init__(self, config: Config, r_squared: pd.Series, outer_model: pd.DataFrame):
        path = config.path()
        lv_type = path.sum(axis=1).astype(bool)
        lv_type.name = "type"
        lv_type_text = lv_type.replace(False, "Exogenous").replace(True, "Endogenous")
        block_communality = pd.Series(0, index=path.index, name="block_communality")
        mean_redundancy = pd.Series(0, index=path.index, name="mean_redundancy")
        ave = pd.Series(0, index=path.index, name="ave")
        communality_aux = []
        num_mvs_in_lv = []
        for lv in config.lvs():
            communality = outer_model.loc[:, "communality"].loc[config.mvs(lv)]
            block_communality.loc[lv] = communality.mean()
            mean_redundancy.loc[lv] = outer_model.loc[:, "redundancy"].loc[config.mvs(lv)].mean()
            if config.mode(lv) == Mode.A:
                ave_numerator = communality.sum()
                ave_denominator = ave_numerator + (1 - communality).sum()
                ave.loc[lv] = ave_numerator / ave_denominator
            if len(config.mvs(lv)) > 1:
                num_mvs_in_lv.append(len(config.mvs(lv)))
                communality_aux.append(block_communality.loc[lv])
        self.__summary = pd.concat([lv_type_text, r_squared, block_communality, mean_redundancy, ave], axis=1,
                                   sort=True)
        mean_communality = sum(x * y for x, y in zip(communality_aux, num_mvs_in_lv)) / sum(num_mvs_in_lv)
        r_squared_aux = r_squared * lv_type
        self.__goodness_of_fit = np.sqrt(mean_communality * r_squared_aux[r_squared_aux != 0].mean())

    def summary(self) -> pd.DataFrame:
        """Internal method that returns the summary of the inner model."""
        return self.__summary

    def goodness_of_fit(self) -> float:
        """Internal method that returns the goodness-of-fit of the model."""
        return self.__goodness_of_fit
