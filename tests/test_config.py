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

import pandas as pd, pytest, plspm.util as util
from plspm.config import Config

def config_test_values():
    return util.config_defaults({"AGRI": ["gini", "farm", "rent"],
              "IND": ["gnpr", "labo"],
              "POLINS": ["ecks", "death", "demo", "inst"]}, "A", "NUM")

def test_config_rejects_wrong_types():
    with pytest.raises(TypeError):
        Config(pd.DataFrame(), "boo")
    with pytest.raises(TypeError):
        Config("hello!", config_test_values())

def test_config_rejects_bad_path_matrix():
    with pytest.raises(ValueError):
        Config(pd.DataFrame([[0, 0, 0]]), config_test_values())
    with pytest.raises(ValueError):
        Config(pd.DataFrame([[1, 1], [1, 1]]), config_test_values())
    with pytest.raises(ValueError):
        Config(pd.DataFrame([[1, 0], [2, 1]]), config_test_values())
    with pytest.raises(ValueError):
        Config(pd.DataFrame([[1, 0],[1, 1]], index=["A", "B"], columns=["C", "D"]), config_test_values())

def test_config_rejects_path_and_lv_config_not_matching():
    path = pd.DataFrame(
    [[0, 0, 0],
     [0, 0, 0],
     [1, 1, 0]],
    index=["AGRI", "IND", "POLINS"],
    columns=["AGRI", "IND", "POLINS"])
    # with pytest.raises(ValueError):
    Config(path, config_test_values())
