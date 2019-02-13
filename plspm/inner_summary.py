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

import pandas as pd

class InnerSummary:

    def __init__(self, path, blocks, inner_model, outer_model):
        lv_type = path.sum(axis=1).astype(bool).replace(False, "Exogenous").replace(True, "Endogenous")
        lv_type.name = "type"
        block_communality = pd.Series(0, index=path.index, name="block_communality")
        mean_redundancy = pd.Series(0, index=path.index, name="mean_redundancy")
        ave = pd.Series(0, index=path.index, name="ave")
        for lv in blocks:
            communality = outer_model.loc[:,"communality"].loc[blocks[lv]]
            block_communality.loc[lv] = communality.mean()
            mean_redundancy.loc[lv] = outer_model.loc[:,"redundancy"].loc[blocks[lv]].mean()
            ave_numerator = communality.sum()
            ave_denominator = ave_numerator + (1 - communality).sum()
            ave.loc[lv] = ave_numerator / ave_denominator
        self.__summary = pd.concat([lv_type, inner_model.r_squared(), block_communality, mean_redundancy, ave], axis=1, sort=True)

    def summary(self):
        return self.__summary
