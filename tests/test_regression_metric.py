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

import pandas.testing as pt, pandas as pd, plspm.util as util, numpy.testing as npt, plspm.config as c, math
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode


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
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    config = c.Config(satisfaction_path_matrix(), scaled=False)
    config.add_lv_with_columns_named("imag", satisfaction, "IMAG", Mode.A)
    config.add_lv_with_columns_named("expe", satisfaction, "EXPE", Mode.A)
    config.add_lv_with_columns_named("qual", satisfaction, "QUAL", Mode.A)
    config.add_lv_with_columns_named("val", satisfaction, "VAL", Mode.A)
    config.add_lv_with_columns_named("sat", satisfaction, "SAT", Mode.A)
    config.add_lv_with_columns_named("loy", satisfaction, "LOY", Mode.A)

    plspm_calc = Plspm(satisfaction, config)
    expected_scores = pd.read_csv("file:tests/data/satisfaction.scores.csv")
    npt.assert_allclose(util.sort_cols(expected_scores), util.sort_cols(plspm_calc.scores()))

    expected_inner_model = pd.read_csv("file:tests/data/satisfaction.inner-model.csv", index_col=0)
    actual_inner_model = plspm_calc.inner_model()["SAT"].drop(['const'])
    npt.assert_allclose(util.sort_cols(expected_inner_model).sort_index(),
                        util.sort_cols(actual_inner_model).sort_index())
    expected_outer_model = pd.read_csv("file:tests/data/satisfaction.outer-model.csv", index_col=0)
    npt.assert_allclose(
        util.sort_cols(expected_outer_model.drop(["block"], axis=1)).sort_index(),
        util.sort_cols(plspm_calc.outer_model()).sort_index())

    expected_crossloadings = pd.read_csv("file:tests/data/satisfaction.crossloadings.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_crossloadings.drop(["block"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.crossloadings()).sort_index())

    expected_inner_summary = pd.read_csv("file:tests/data/satisfaction.inner-summary.csv", index_col=0)

    npt.assert_allclose(util.sort_cols(expected_inner_summary.drop(["type"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.inner_summary().drop(["type"], axis=1)).sort_index())
    pt.assert_series_equal(expected_inner_summary.loc[:, "type"].sort_index(),
                           plspm_calc.inner_summary().loc[:, "type"].sort_index())

    expected_effects = pd.read_csv("file:tests/data/satisfaction.effects.csv")

    pt.assert_frame_equal(expected_effects.loc[:, ["from", "to"]],
                           plspm_calc.effects().loc[:, ["from", "to"]])
    npt.assert_allclose(expected_effects.drop(["from", "to"], axis=1),
                        plspm_calc.effects().drop(["from", "to"], axis=1))

    assert math.isclose(0.609741624338411,plspm_calc.goodness_of_fit())

    plspm_calc_path = Plspm(satisfaction, config, Scheme.PATH)
    expected_outer_model_path = util.sort_cols(
        pd.read_csv("file:tests/data/satisfaction.outer-model-path.csv", index_col=0).drop(["block"],
                                                                                           axis=1)).sort_index()
    npt.assert_allclose(expected_outer_model_path,
                        util.sort_cols(plspm_calc_path.outer_model()).sort_index())

    plspm_calc_factorial = Plspm(satisfaction, config, Scheme.FACTORIAL)
    expected_outer_model_factorial = util.sort_cols(
        pd.read_csv("file:tests/data/satisfaction.outer-model-factorial.csv", index_col=0).drop(["block"],
                                                                                                axis=1)).sort_index()
    npt.assert_allclose(expected_outer_model_factorial,
                        util.sort_cols(plspm_calc_factorial.outer_model()).sort_index())

def test_plspm_russa_mode_b():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    config = c.Config(satisfaction_path_matrix(), scaled=False)
    config.add_lv_with_columns_named("imag", satisfaction, "IMAG", Mode.B)
    config.add_lv_with_columns_named("expe", satisfaction, "EXPE", Mode.B)
    config.add_lv_with_columns_named("qual", satisfaction, "QUAL", Mode.B)
    config.add_lv_with_columns_named("val", satisfaction, "VAL", Mode.B)
    config.add_lv_with_columns_named("sat", satisfaction, "SAT", Mode.B)
    config.add_lv_with_columns_named("loy", satisfaction, "LOY", Mode.B)

    plspm_calc = Plspm(satisfaction, config, Scheme.CENTROID)
    print(plspm_calc.inner_summary())
    expected_inner_summary = pd.read_csv("file:tests/data/satisfaction.modeb.inner-summary.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_inner_summary.drop(["type"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.inner_summary().drop(["type"], axis=1)).sort_index())
    pt.assert_series_equal(expected_inner_summary.loc[:, "type"].sort_index(),
                           plspm_calc.inner_summary().loc[:, "type"].sort_index())
