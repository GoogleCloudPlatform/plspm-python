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

import pytest, pandas.testing as pt, plspm.config as c
from plspm.mode import Mode
from plspm.estimator import Estimator

def test_can_add_hoc_lv_paths_correctly():
    structure = c.Structure()
    structure.add_path(["MANDRILL", "BONOBO"], ["APE"])
    structure.add_path(["APE"], ["GOAT"])
    initial_path = structure.path()
    config = c.Config(initial_path)
    config.add_higher_order("APE", Mode.A, ["CHEDDAR", "GOUDA"])
    estimator = Estimator(config)
    structure = c.Structure(initial_path)
    structure.add_path(["MANDRILL", "BONOBO"], ["CHEDDAR"])
    structure.add_path(["MANDRILL", "BONOBO"], ["GOUDA"])
    structure.add_path(["GOUDA", "CHEDDAR"], ["GOAT"])
    expected = structure.path().drop("APE").drop("APE", axis=1)
    actual = estimator.hoc_path_first_stage(config)
    pt.assert_frame_equal(expected, actual)