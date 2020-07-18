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

import pandas.testing as pt, pandas as pd, plspm.util as util, numpy.testing as npt, plspm.config as c, math, pytest
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode

def satisfaction_path_matrix():
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    return structure.path()


def test_plspm_satisfaction():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    config = c.Config(satisfaction_path_matrix(), scaled=False)
    config.add_lv_with_columns_named("IMAG", Mode.A, satisfaction, "imag")
    config.add_lv_with_columns_named("EXPE", Mode.A, satisfaction, "expe")
    config.add_lv_with_columns_named("VAL", Mode.A, satisfaction, "val")
    config.add_lv_with_columns_named("QUAL", Mode.A, satisfaction, "qual")
    config.add_lv_with_columns_named("SAT", Mode.A, satisfaction, "sat")
    config.add_lv_with_columns_named("LOY", Mode.A, satisfaction, "loy")

    plspm_calc = Plspm(satisfaction, config)
    expected_scores = pd.read_csv("file:tests/data/satisfaction.scores.csv")
    npt.assert_allclose(util.sort_cols(expected_scores), util.sort_cols(plspm_calc.scores()))

    expected_inner_model = pd.read_csv("file:tests/data/satisfaction.inner-model.csv", index_col=0)
    actual_inner_model = plspm_calc.inner_model()
    actual_inner_model = actual_inner_model[actual_inner_model['to'].isin(["SAT"])].drop(["to"], axis=1)
    npt.assert_allclose(util.sort_cols(expected_inner_model).sort_index(),
                        util.sort_cols(actual_inner_model.set_index(["from"],drop=True)).sort_index())
    expected_outer_model = pd.read_csv("file:tests/data/satisfaction.outer-model.csv", index_col=0).drop(["block"], axis=1)
    pt.assert_index_equal(expected_outer_model.columns, plspm_calc.outer_model().columns)
    npt.assert_allclose(
        util.sort_cols(expected_outer_model.sort_index()),
        util.sort_cols(plspm_calc.outer_model()).sort_index())

    expected_crossloadings = pd.read_csv("file:tests/data/satisfaction.crossloadings.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_crossloadings.drop(["block"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.crossloadings()).sort_index())

    expected_inner_summary = pd.read_csv("file:tests/data/satisfaction.inner-summary.csv", index_col=0)

    npt.assert_allclose(util.sort_cols(expected_inner_summary.drop(["type"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.inner_summary().drop(["type", "r_squared_adj"], axis=1)).sort_index())
    pt.assert_series_equal(expected_inner_summary.loc[:, "type"].sort_index(),
                           plspm_calc.inner_summary().loc[:, "type"].sort_index())

    expected_effects = pd.read_csv("file:tests/data/satisfaction.effects.csv", index_col=0)

    pt.assert_frame_equal(expected_effects.loc[:, ["from", "to"]].sort_index(),
                           plspm_calc.effects().loc[:, ["from", "to"]].sort_index())
    npt.assert_allclose(expected_effects.drop(["from", "to"], axis=1).sort_index(),
                        plspm_calc.effects().drop(["from", "to"], axis=1).sort_index())

    expected_unidimensionality = pd.read_csv("file:tests/data/satisfaction_unidim.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_unidimensionality.drop(["mode"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.unidimensionality().drop(["mode"], axis=1)).sort_index())

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
    config.add_lv_with_columns_named("QUAL", Mode.B, satisfaction, "qual")
    config.add_lv_with_columns_named("VAL", Mode.B, satisfaction, "val")
    config.add_lv_with_columns_named("SAT", Mode.B, satisfaction, "sat")
    config.add_lv_with_columns_named("LOY", Mode.B, satisfaction, "loy")
    config.add_lv_with_columns_named("IMAG", Mode.B, satisfaction, "imag")
    config.add_lv_with_columns_named("EXPE", Mode.B, satisfaction, "expe")

    plspm_calc = Plspm(satisfaction, config, Scheme.CENTROID)
    expected_inner_summary = pd.read_csv("file:tests/data/satisfaction.modeb.inner-summary.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_inner_summary.drop(["type"], axis=1)).sort_index(),
                        util.sort_cols(plspm_calc.inner_summary().drop(["type", "r_squared_adj"], axis=1)).sort_index())
    pt.assert_series_equal(expected_inner_summary.loc[:, "type"].sort_index(),
                           plspm_calc.inner_summary().loc[:, "type"].sort_index())

def test_only_single_item_constructs():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    config = c.Config(satisfaction_path_matrix())
    config.add_lv("QUAL", Mode.A, c.MV("qual1"))
    config.add_lv("VAL", Mode.A, c.MV("val1"))
    config.add_lv("SAT", Mode.A, c.MV("sat1"))
    config.add_lv("LOY", Mode.A, c.MV("loy1"))
    config.add_lv("IMAG", Mode.A, c.MV("imag1"))
    config.add_lv("EXPE", Mode.A, c.MV("expe1"))

    plspm_calc = Plspm(satisfaction, config, Scheme.CENTROID)
    with pytest.raises(ValueError):
        plspm_calc.goodness_of_fit()
        