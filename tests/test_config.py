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

import pandas as pd, pytest, numpy.testing as npt, plspm.mode as mode, plspm.config as c


def config_test_path_matrix():
    lvs = ["AGRI", "IND", "POLINS"]
    return pd.DataFrame(
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 0]],
        index=lvs, columns=lvs)


def test_config_rejects_bad_path_matrix():
    # Takes a matrix
    with pytest.raises(TypeError):
        c.Config("hello")
    # Matrix should be square
    with pytest.raises(ValueError):
        c.Config(pd.DataFrame([[0, 0, 0]]))
    # Matrix should be lower triangular
    with pytest.raises(ValueError):
        c.Config(pd.DataFrame([[1, 1], [1, 1]]))
    # Only 1 and 0 allowed in matrix
    with pytest.raises(ValueError):
        c.Config(pd.DataFrame([[1, 0], [2, 1]]))
    # Indices and columns should have the same names
    with pytest.raises(ValueError):
        c.Config(pd.DataFrame([[1, 0], [1, 1]], index=["A", "B"], columns=["C", "D"]))


def test_config_rejects_path_and_lv_config_not_matching():
    config = c.Config(config_test_path_matrix())
    with pytest.raises(ValueError):
        config.add_lv("POO", mode.A, c.MV("test"))


def test_config_returns_correct_mode_and_mvs():
    config = c.Config(config_test_path_matrix())
    config.add_lv("AGRI", mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    assert config.mode("AGRI") == mode.A
    npt.assert_array_equal(config.blocks()["AGRI"], ["gini", "farm", "rent"])


def test_config_rejects_missing_mvs():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(config_test_path_matrix())
    config.add_lv("AGRI", mode.A, c.MV("gini"), c.MV("farm"), c.MV("poo"))
    with pytest.raises(ValueError):
        config.filter(russa)

def test_config_filters_mvs():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(config_test_path_matrix())
    config.add_lv("AGRI", mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    npt.assert_array_equal(list(config.filter(russa)), ["gini", "farm", "rent"])
