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

import pandas.testing as pt, pandas as pd, plspm.scheme as scheme, plspm.util as util, numpy.testing as npt
from plspm.plspm import Plspm

def test_plspm_russa():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    rus_path = pd.DataFrame(
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 0]],
        index=["AGRI", "IND", "POLINS"],
        columns=["AGRI", "IND", "POLINS"])
    rus_blocks = util.config_defaults({"AGRI": ["gini", "farm", "rent"],
                  "IND": ["gnpr", "labo"],
                  "POLINS": ["ecks", "death", "demo", "inst"]}, "A", "NUM")

    plspm_calc = Plspm(russa, rus_path, rus_blocks, scheme.CENTROID, 100, 0.0000001)
    expected_scores = pd.read_csv("file:tests/data/russa.scores.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_scores), util.sort_cols(plspm_calc.scores()))

    expected_inner_model = pd.read_csv("file:tests/data/russa.inner_model.csv", index_col=0)
    actual_inner_model = plspm_calc.inner_model()["POLINS"].drop(['const'])
    npt.assert_allclose(util.sort_cols(expected_inner_model).sort_index(), util.sort_cols(actual_inner_model).sort_index())

    expected_outer_model = pd.read_csv("file:tests/data/russa.outer_model.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_outer_model.filter(["weight", "loading", "communality", "redundancy"])).sort_index(),
                        util.sort_cols(plspm_calc.outer_model()).sort_index())

    expected_crossloadings = pd.read_csv("file:tests/data/russa.crossloadings.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_crossloadings.filter(["AGRI", "IND", "POLINS"])).sort_index(), util.sort_cols(plspm_calc.crossloadings()).sort_index())

    expected_inner_summary = pd.read_csv("file:tests/data/russa.inner_summary.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_inner_summary.filter(["r_squared", "block_communality", "mean_redundancy", "ave"])).sort_index(),
                        util.sort_cols(plspm_calc.inner_summary().filter(["r_squared", "block_communality", "mean_redundancy", "ave"])).sort_index())
    pt.assert_series_equal(expected_inner_summary.loc[:,"type"].sort_index(), plspm_calc.inner_summary().loc[:,"type"].sort_index())

    plspm_calc_path = Plspm(russa, rus_path, rus_blocks, scheme.PATH, 100, 0.0000001)
    expected_outer_model_path = util.sort_cols(pd.read_csv("file:tests/data/russa.outer_model_path.csv", index_col=0).filter(["weight", "loading", "communality", "redundancy"])).sort_index()
    npt.assert_allclose(expected_outer_model_path,
                        util.sort_cols(plspm_calc_path.outer_model()).sort_index())

    plspm_calc_factorial = Plspm(russa, rus_path, rus_blocks, scheme.FACTORIAL, 100, 0.0000001)
    expected_outer_model_factorial = util.sort_cols(pd.read_csv("file:tests/data/russa.outer_model_factorial.csv", index_col=0).filter(["weight", "loading", "communality", "redundancy"])).sort_index()
    npt.assert_allclose(expected_outer_model_factorial,
                        util.sort_cols(plspm_calc_factorial.outer_model()).sort_index())
