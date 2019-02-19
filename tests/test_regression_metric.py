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

import pandas.testing as pt, pandas as pd, plspm.scheme as scheme, plspm.util as util, numpy.testing as npt, \
    plspm.mode as mode, plspm.config as c
from plspm.plspm import Plspm


def satisfaction_path_matrix():
    lvs = ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]
    return pd.DataFrame(
        [[0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 0, 0],
         [1, 0, 0, 0, 1, 0],
         ],
        index=lvs, columns=lvs)


def test_plspm_satisfaction():
    russa = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    config = c.Config(satisfaction_path_matrix(), scaled=False)
    config.add_lv("IMAG", mode.A, c.MV("imag1"), c.MV("imag2"), c.MV("imag3"), c.MV("imag4"), c.MV("imag5"))
    config.add_lv("EXPE", mode.A, c.MV("expe1"), c.MV("expe2"), c.MV("expe3"), c.MV("expe4"), c.MV("expe5"))
    config.add_lv("QUAL", mode.A, c.MV("qual1"), c.MV("qual2"), c.MV("qual3"), c.MV("qual4"), c.MV("qual5"))
    config.add_lv("VAL", mode.A, c.MV("val1"), c.MV("val2"), c.MV("val3"), c.MV("val4"))
    config.add_lv("SAT", mode.A, c.MV("sat1"), c.MV("sat2"), c.MV("sat3"), c.MV("sat4"))
    config.add_lv("LOY", mode.A, c.MV("loy1"), c.MV("loy2"), c.MV("loy3"), c.MV("loy4"))

    plspm_calc = Plspm(russa, config)
    expected_scores = pd.read_csv("file:tests/data/satisfaction.scores.csv")
    npt.assert_allclose(util.sort_cols(expected_scores), util.sort_cols(plspm_calc.scores()))

    expected_inner_model = pd.read_csv("file:tests/data/satisfaction.inner-model.csv", index_col=0)
    actual_inner_model = plspm_calc.inner_model()["SAT"].drop(['const'])
    npt.assert_allclose(util.sort_cols(expected_inner_model).sort_index(),
                        util.sort_cols(actual_inner_model).sort_index())
    expected_outer_model = pd.read_csv("file:tests/data/satisfaction.outer-model.csv", index_col=0)
    print(plspm_calc.outer_model())
    npt.assert_allclose(
        util.sort_cols(expected_outer_model.filter(["weight", "loading", "communality", "redundancy"])).sort_index(),
        util.sort_cols(plspm_calc.outer_model()).sort_index())

#    expected_crossloadings = pd.read_csv("file:tests/data/russa.crossloadings.csv", index_col=0)
#    npt.assert_allclose(util.sort_cols(expected_crossloadings.filter(["AGRI", "IND", "POLINS"])).sort_index(),
#                        util.sort_cols(plspm_calc.crossloadings()).sort_index())

#    expected_inner_summary = pd.read_csv("file:tests/data/russa.inner_summary.csv", index_col=0)
#    npt.assert_allclose(util.sort_cols(
#        expected_inner_summary.filter(["r_squared", "block_communality", "mean_redundancy", "ave"])).sort_index(),
#                        util.sort_cols(plspm_calc.inner_summary().filter(
#                            ["r_squared", "block_communality", "mean_redundancy", "ave"])).sort_index())
#    pt.assert_series_equal(expected_inner_summary.loc[:, "type"].sort_index(),
#                           plspm_calc.inner_summary().loc[:, "type"].sort_index())

#    plspm_calc_path = Plspm(russa, config, scheme.PATH, 100, 0.0000001)
#    expected_outer_model_path = util.sort_cols(
#        pd.read_csv("file:tests/data/russa.outer_model_path.csv", index_col=0).filter(
#            ["weight", "loading", "communality", "redundancy"])).sort_index()
#    npt.assert_allclose(expected_outer_model_path,
#                        util.sort_cols(plspm_calc_path.outer_model()).sort_index())

#    plspm_calc_factorial = Plspm(russa, config, scheme.FACTORIAL, 100, 0.0000001)
#    expected_outer_model_factorial = util.sort_cols(
#        pd.read_csv("file:tests/data/russa.outer_model_factorial.csv", index_col=0).filter(
#            ["weight", "loading", "communality", "redundancy"])).sort_index()
#    npt.assert_allclose(expected_outer_model_factorial,
#                        util.sort_cols(plspm_calc_factorial.outer_model()).sort_index())
