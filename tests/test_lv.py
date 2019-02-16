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

import plspm.weights as ow, pandas as pd, plspm.util as util
from plspm.config import Config

def test_mv_should_be_grouped_by_lv():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    rus_path = pd.DataFrame(
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 0]],
        index=["AGRI", "IND", "POLINS"],
        columns=["AGRI", "IND", "POLINS"])
    rus_blocks = util.config_defaults({"AGRI": ["gini", "farm", "rent"],
                  "IND": ["gnpr", "labo"],
                  "POLINS": ["ecks", "death", "demo"]}, "A", "NUM")
    ow_calculator = ow.Weights(russa, Config(rus_path, rus_blocks))
    mv_grouped_by_lv = ow_calculator.mv_grouped_by_lv()
    assert set(["AGRI", "IND", "POLINS"]) == set(mv_grouped_by_lv.keys())
    assert set(["gini", "farm", "rent"]) == set(mv_grouped_by_lv["AGRI"])
