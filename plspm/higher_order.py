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

class HOCEstimator:
    def __init__(self, config: c.Config):
        self.__config = config

    def hoc_weights(self, data: pd.DataFrame):
        hocs = self.__config.hoc()
        for hoc in hocs:
            new_mvs = []
            for lv in hocs[hoc]:
                mvs = self.__config.mvs(lv)
                for mv in mvs:
                    mv_new = hoc + "_" + mv
                    data[mv_new] = data[mv]
                    new_mvs.append(c.MV(mv_new, self.__config.scale(mv)))
                if lv not in list(self.__config.path()):
                    self.__config.remove_lv(lv)
            self.__config.add_lv(hoc, self.__config.mode(hoc), *new_mvs)
        return data
                            
