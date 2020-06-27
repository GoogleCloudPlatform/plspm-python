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

import pandas.testing as pt, pandas as pd, plspm.util as util, numpy.testing as npt, plspm.config as c, math, numpy as np
from plspm.plspm import Plspm
from plspm.scale import Scale
from plspm.scheme import Scheme
from plspm.mode import Mode


def russa_path_matrix():
    structure = c.Structure()
    structure.add_path(["AGRI", "IND"], ["POLINS"])
    return structure.path()

def test_plspm_russa():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(russa_path_matrix(), default_scale=Scale.NUM)
    config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("rent"), c.MV("farm"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))

    plspm_calc = Plspm(russa, config, Scheme.CENTROID, 100, 0.0000001)
    expected_scores = pd.read_csv("file:tests/data/russa.scores.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_scores), util.sort_cols(plspm_calc.scores()))

    expected_inner_model = pd.read_csv("file:tests/data/russa.inner_model.csv", index_col=0)
    actual_inner_model = plspm_calc.inner_model()
    actual_inner_model = actual_inner_model[actual_inner_model['to'].isin(["POLINS"])].drop(["to"], axis=1)

    npt.assert_allclose(util.sort_cols(expected_inner_model).sort_index(),
                        util.sort_cols(actual_inner_model.set_index(["from"],drop=True)).sort_index())

    expected_outer_model = pd.read_csv("file:tests/data/russa.outer_model.csv", index_col=0)
    npt.assert_allclose(
        util.sort_cols(expected_outer_model.filter(["weight", "loading", "communality", "redundancy"])).sort_index(),
        util.sort_cols(plspm_calc.outer_model()).sort_index())

    expected_crossloadings = pd.read_csv("file:tests/data/russa.crossloadings.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_crossloadings.filter(["AGRI", "IND", "POLINS"])).sort_index(),
                        util.sort_cols(plspm_calc.crossloadings()).sort_index())

    expected_inner_summary = pd.read_csv("file:tests/data/russa.inner_summary.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_inner_summary.drop(["type"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.inner_summary().drop(["type", "r_squared_adj"], axis=1)).sort_index())
    pt.assert_series_equal(expected_inner_summary.loc[:, "type"].sort_index(),
                           plspm_calc.inner_summary().loc[:, "type"].sort_index())

    assert math.isclose(0.643594505232204, plspm_calc.goodness_of_fit())

    plspm_calc_path = Plspm(russa, config, Scheme.PATH, 100, 0.0000001)
    expected_outer_model_path = util.sort_cols(
        pd.read_csv("file:tests/data/russa.outer_model_path.csv", index_col=0).filter(
            ["weight", "loading", "communality", "redundancy"])).sort_index()
    npt.assert_allclose(expected_outer_model_path,
                        util.sort_cols(plspm_calc_path.outer_model()).sort_index())

    plspm_calc_factorial = Plspm(russa, config, Scheme.FACTORIAL, 100, 0.0000001)
    expected_outer_model_factorial = util.sort_cols(
        pd.read_csv("file:tests/data/russa.outer_model_factorial.csv", index_col=0).filter(
            ["weight", "loading", "communality", "redundancy"])).sort_index()
    npt.assert_allclose(expected_outer_model_factorial,
                        util.sort_cols(plspm_calc_factorial.outer_model()).sort_index())


def test_plspm_russa_mode_b():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(russa_path_matrix(), default_scale=Scale.NUM)
    config.add_lv("AGRI", Mode.B, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("POLINS", Mode.B, c.MV("ecks"), c.MV("demo"), c.MV("inst"), c.MV("death"))
    config.add_lv("IND", Mode.B, c.MV("gnpr"), c.MV("labo"))

    plspm_calc = Plspm(russa, config, Scheme.CENTROID, 100, 0.0000001)
    expected_inner_summary = pd.read_csv("file:tests/data/russa.mode_b_inner_summary.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_inner_summary.drop(["type"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.inner_summary().drop(["type", "r_squared_adj"], axis=1)).sort_index())
    pt.assert_series_equal(expected_inner_summary.loc[:, "type"].sort_index(),
                           plspm_calc.inner_summary().loc[:, "type"].sort_index())

def test_plspm_russa_categorical():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(russa_path_matrix(), default_scale=Scale.NUM)
    config.add_lv("IND", Mode.A, c.MV("gnpr", Scale.ORD), c.MV("labo", Scale.ORD))
    config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo", Scale.NOM), c.MV("inst"))
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))

    plspm_calc = Plspm(russa, config, Scheme.CENTROID, 100, 0.0000001)
    expected_inner_summary = pd.read_csv("file:tests/data/russa.categorical.inner_summary.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_inner_summary.drop(["type"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.inner_summary().drop(["type", "r_squared_adj"], axis=1)).sort_index())
    pt.assert_series_equal(expected_inner_summary.loc[:, "type"].sort_index(),
                           plspm_calc.inner_summary().loc[:, "type"].sort_index())

def test_plspm_russa_categorical_mode_b():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(russa_path_matrix(), default_scale=Scale.NUM)
    config.add_lv("AGRI", Mode.B, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.B, c.MV("gnpr", Scale.ORD), c.MV("labo", Scale.ORD))
    config.add_lv("POLINS", Mode.B, c.MV("ecks"), c.MV("death"), c.MV("demo", Scale.NOM), c.MV("inst"))

    plspm_calc = Plspm(russa, config, Scheme.CENTROID, 100, 0.0000001)
    expected_inner_summary = pd.read_csv("file:tests/data/russa.categorical.mode_b.inner_summary.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_inner_summary.drop(["type"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.inner_summary().drop(["type", "r_squared_adj"], axis=1)).sort_index())
    pt.assert_series_equal(expected_inner_summary.loc[:, "type"].sort_index(),
                           plspm_calc.inner_summary().loc[:, "type"].sort_index())

def test_plspm_russa_missing_data():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    russa.iloc[0, 0] = np.NaN
    russa.iloc[3, 3] = np.NaN
    russa.iloc[5, 5] = np.NaN
    config = c.Config(russa_path_matrix(), default_scale=Scale.NUM)
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))

    plspm_calc = Plspm(russa, config, Scheme.CENTROID, 100, 0.0000001)
    expected_inner_summary = pd.read_csv("file:tests/data/russa.missing.inner_summary.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_inner_summary.drop(["type"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.inner_summary().drop(["type", "r_squared_adj"], axis=1)).sort_index())
    pt.assert_series_equal(expected_inner_summary.loc[:, "type"].sort_index(),
                           plspm_calc.inner_summary().loc[:, "type"].sort_index())
    assert plspm_calc.unidimensionality().drop(["mode", "mvs"], axis=1).isnull().values.all()
