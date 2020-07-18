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

import pandas as pd, pytest, numpy.testing as npt, plspm.config as c, pandas.testing as pt
from plspm.scale import Scale
from plspm.mode import Mode


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


def test_config_rejects_adding_lv_not_present_in_path():
    config = c.Config(config_test_path_matrix())
    with pytest.raises(ValueError):
        config.add_lv("POO", Mode.A, c.MV("test"))


def test_config_rejects_path_and_lv_config_not_matching():
    config = c.Config(config_test_path_matrix())
    config.add_lv("AGRI", Mode.A, c.MV("gini"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"))
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    with pytest.raises(ValueError):
        config.filter(russa)


def test_config_returns_correct_mode_and_mvs():
    config = c.Config(config_test_path_matrix())
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    assert config.mode("AGRI") == Mode.A
    npt.assert_array_equal(config.mvs("AGRI"), ["gini", "farm", "rent"])


def test_config_rejects_missing_mvs():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(config_test_path_matrix())
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("poo"))
    with pytest.raises(ValueError):
        config.filter(russa)

def test_config_filters_mvs():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(config_test_path_matrix())
    config.add_lv("POLINS", Mode.A)
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.A)
    npt.assert_array_equal(list(config.filter(russa)), ["gini", "farm", "rent"])

def test_data_should_only_contain_numerical_values():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    russa['gini'] = russa['gini'].astype(str)
    config = c.Config(config_test_path_matrix())
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    with pytest.raises(ValueError):
        config.filter(russa)

def test_all_mvs_should_have_a_scale_if_data_is_nonmetric():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(config_test_path_matrix())
    config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))
    config.add_lv("AGRI", Mode.A, c.MV("gini", Scale.NUM), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    with pytest.raises(TypeError):
        config.treat(config.filter(russa))

def test_scaling_should_be_false_if_all_raw():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(config_test_path_matrix(), default_scale=Scale.RAW)
    config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    config.treat(config.filter(russa))
    assert not config.scaled()

def test_scaling_should_be_true_and_all_scales_set_to_num_if_only_raw_and_num_supplied():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(config_test_path_matrix(), default_scale=Scale.RAW)
    config.add_lv("POLINS", Mode.A, c.MV("ecks", Scale.NUM), c.MV("death"), c.MV("demo"), c.MV("inst"))
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    config.treat(config.filter(russa))
    assert config.scaled()
    for mv in ["gini", "farm", "rent"]:
        assert config.scale(mv) == Scale.NUM

def test_scales_should_remain_unchanged_if_values_other_than_num_and_raw_supplied():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(config_test_path_matrix(), default_scale=Scale.RAW, scaled=False)
    config.add_lv("AGRI", Mode.A, c.MV("gini", Scale.NUM), c.MV("farm", Scale.ORD), c.MV("rent"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))
    config.filter(russa)
    assert not config.scaled()
    assert config.scale("farm") == Scale.ORD


def test_create_path_from_structure():
    lvs = ["MANDRILL", "BONOBO", "APE", "GOAT", "CATFISH"]
    expected = pd.DataFrame(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
        index=lvs, columns=lvs)
    structure = c.Structure()
    structure.add_path(source=["BONOBO", "MANDRILL"], target=["APE"])
    structure.add_path(source=["APE"], target=["CATFISH", "GOAT"])
    pt.assert_frame_equal(expected, structure.path())

def test_cannot_add_mvs_twice():
    structure = c.Structure()
    structure.add_path(source=["BONOBO"], target=["APE"])
    config = c.Config(structure.path())
    config.add_lv("BONOBO", Mode.A, c.MV("a"), c.MV("b"))
    with pytest.raises(ValueError):
        config.add_lv("APE", Mode.A, c.MV("a"), c.MV("b"))

def test_cannot_add_mv_with_same_name_as_lv():
    structure = c.Structure()
    structure.add_path(source=["BONOBO"], target=["APE"])
    config = c.Config(structure.path())
    with pytest.raises(ValueError):
        config.add_lv("BONOBO", Mode.A, c.MV("BONOBO"))
