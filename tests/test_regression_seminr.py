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
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

def test_paths():
    mobi = pd.read_csv("file:tests/data/mobi.csv", index_col=0)

    structure = c.Structure()
    structure.add_path(["Expectation", "Quality"], ["Loyalty"])
    structure.add_path(["Image"], ["Expectation"])
    structure.add_path(["Complaints"], ["Loyalty"])

    config = c.Config(structure.path(), default_scale=Scale.NUM)
    config.add_lv_with_columns_named("Expectation", Mode.A, mobi, "CUEX")
    config.add_lv_with_columns_named("Quality", Mode.B, mobi, "PERQ")
    config.add_lv_with_columns_named("Loyalty", Mode.A, mobi, "CUSL")
    config.add_lv_with_columns_named("Image", Mode.A, mobi, "IMAG")
    config.add_lv_with_columns_named("Complaints", Mode.A, mobi, "CUSCO")
    mobi_pls = Plspm(mobi, config, Scheme.PATH, 100, 0.00000001)
    expected_outer_model = pd.read_csv("file:tests/data/seminr-mobi-basic-outer-model.csv", index_col=0)
    actual_outer_model = mobi_pls.outer_model().drop(["communality","redundancy"], axis=1)
    npt.assert_allclose(expected_outer_model.sort_index(), actual_outer_model.sort_index(), rtol=1e-5)

    expected_paths = pd.read_csv("file:tests/data/seminr-mobi-basic-paths.csv", index_col=0)
    actual_paths = mobi_pls.path_coefficients().transpose()
    npt.assert_allclose(expected_paths.sort_index().sort_index(axis=1), actual_paths.sort_index().sort_index(axis=1), rtol=1e-6)

def test_hoc_two_stage():
    mobi = pd.read_csv("file:tests/data/mobi.csv", index_col=0)

    structure = c.Structure()
    structure.add_path(["Expectation", "Quality"], ["Satisfaction"])
    structure.add_path(["Satisfaction"], ["Complaints", "Loyalty"])

    config = c.Config(structure.path(), default_scale=Scale.NUM)
    config.add_higher_order("Satisfaction", Mode.A, ["Image", "Value"])
    config.add_lv_with_columns_named("Expectation", Mode.A, mobi, "CUEX")
    config.add_lv_with_columns_named("Quality", Mode.B, mobi, "PERQ")
    config.add_lv_with_columns_named("Loyalty", Mode.A, mobi, "CUSL")
    config.add_lv_with_columns_named("Image", Mode.A, mobi, "IMAG")
    config.add_lv_with_columns_named("Complaints", Mode.A, mobi, "CUSCO")
    config.add_lv_with_columns_named("Value", Mode.A, mobi, "PERV")
    mobi_pls = Plspm(mobi, config, Scheme.PATH, 100, 0.00000001)
    expected_outer_model = pd.read_csv("file:tests/data/seminr-mobi-hoc-ts-outer-model.csv", index_col=0)
    actual_outer_model = mobi_pls.outer_model().drop(["communality","redundancy"], axis=1)
    indices = list(set(expected_outer_model.index.values.tolist()).intersection(set(actual_outer_model.index.values.tolist())))
    expected_outer_model = expected_outer_model.loc[indices].sort_index().sort_index(axis=1)
    actual_outer_model = actual_outer_model.loc[indices].sort_index().sort_index(axis=1)
    npt.assert_allclose(expected_outer_model, actual_outer_model, rtol=1e-4)

    expected_paths = pd.read_csv("file:tests/data/seminr-mobi-hoc-ts-paths.csv", index_col=0).transpose()
    actual_paths = mobi_pls.path_coefficients()
    npt.assert_allclose(expected_paths.sort_index().sort_index(axis=1), actual_paths.sort_index().sort_index(axis=1), rtol=1e-6)
