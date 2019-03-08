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

import pandas as pd, numpy as np, numpy.testing as npt, plspm.util as util


def test_impute_missing_values():
    input = pd.DataFrame(
        {"a": [1, 2, np.NaN, 3, np.NaN],
         "b": [1, 2, 3, 4, 5],
         "c": [1, np.NaN, 3, 0, 4]})
    expected_output = pd.DataFrame(
        {"a": [1, 2, 2, 3, 2],
         "b": [1, 2, 3, 4, 5],
         "c": [1, 2, 3, 0, 4]})
    npt.assert_array_equal(expected_output, util.impute(input))

def test_ranking():
    data = pd.Series([0.75, -1.5, 3, -1.5, 15])
    expected_rank = pd.Series([2, 1, 3, 1, 4])
    assert util.rank(data).astype(int).equals(expected_rank)
