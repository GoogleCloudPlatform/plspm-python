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

import plspm.config as c, pandas as pd
from plspm.weights import WeightsCalculatorFactory
from plspm.scale import Scale
from typing import Tuple

class Estimator:
    def __init__(self, config: c.Config):
        self.__config = config

    def prepare_data(self, data: pd.DataFrame):
        hocs = self.__config.hoc()
        for hoc in hocs:
            new_mvs = []
            # create new LV representing HOC combining MVs of constituent first-order construct
            for lv in hocs[hoc]:
                mvs = self.__config.mvs(lv)
                for mv in mvs:
                    mv_new = hoc + "_" + mv
                    data[mv_new] = data[mv]
                    new_mvs.append(c.MV(mv_new, self.__config.scale(mv)))
            self.__config.add_lv(hoc, self.__config.mode(hoc), *new_mvs)
        return data

    def estimate(self, calculator: WeightsCalculatorFactory, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        path = self.first_stage_path()
        treated_data = self.__config.treat(data)
        final_data, scores, weights = calculator.calculate(treated_data, path)

        hocs = self.__config.hoc()
        if None is not None:
            scale = None if self.__config.metric else Scale.NUM
            for hoc in hocs:
                new_mvs = []
                cols_to_drop = [col for col in list(data) if col.startswith(hoc + "_")]
                data = data.drop(columns=cols_to_drop)
                for lv in hocs[hoc]:
                    mv_new = hoc + "_" + lv
                    data[mv_new] = scores[lv]
                    new_mvs.append(c.MV(mv_new, scale))
                self.__config.add_lv(hoc, self.__config.mode(hoc), *new_mvs)
            treated_data = self.__config.treat(data)
            final_data, scores, weights = calculator.calculate(treated_data, self.__config.path())

        return final_data, scores, weights

    def first_stage_path(self) -> pd.DataFrame:
        # For first pass, for HOCs we'll create paths from each and for each exogenous LV to the cconstituent LVs,
        # and from each consituent LV to the HOC.
        path = self.__config.path()
        structure = c.Structure(path)
        for hoc, lvs in self.__config.hoc().items():
            structure.add_path(lvs, [hoc])
            exogenous = path.loc[hoc]
            for lv in list(exogenous[exogenous == 1].index):
                structure.add_path([lv], lvs)
        return structure.path()
            
            
