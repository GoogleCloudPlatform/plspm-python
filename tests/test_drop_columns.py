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

from plspm.drop_columns import DropColumns
import pandas as pd, numpy as np

def test_should_drop_columns():
    input = pd.DataFrame(
        {"a" : [1, 2, 3],
         "b": [1, 2, 3],
         "c": [1, 2, 3],
         "e": [1, 2, 3],
         "d": [1, 2, 3]}
    )
    column_filter = DropColumns(["a", "b"],["c", "d"], input)
    expected_output = input.drop(["e"], axis=1)
    assert expected_output.equals(column_filter.filter());

def test_should_include_mv_columns_that_are_part_of_iv():
    input = pd.DataFrame(
        {"a" : [1, 2, 3],
         "b1": [1, 2, 3],
         "b2": [1, 2, 3],
         "c": [1, 2, 3],
         "e": [1, 2, 3],
         "d": [1, 2, 3]}
    )
    column_filter = DropColumns(["a", "b"],["c", "d"], input)
    expected_output = input.drop(["e"], axis=1)
    assert expected_output.equals(column_filter.filter());

def test_should_not_try_matching_dv_columns():
    input = pd.DataFrame(
        {"a" : [1, 2, 3],
         "b": [1, 2, 3],
         "c": [1, 2, 3],
         "e": [1, 2, 3],
         "d": [1, 2, 3],
         "d1": [1, 2, 3]}
    )
    column_filter = DropColumns(["a", "b"],["c", "d"], input)
    expected_output = input.drop(["e", "d1"], axis=1)
    assert expected_output.equals(column_filter.filter());

def test_should_give_list_of_iv_columns():
    input = pd.DataFrame(
        {"a" : [1, 2, 3],
         "b1": [1, 2, 3],
         "b2": [1, 2, 3],
         "e": [1, 2, 3],
         "d": [1, 2, 3]}
    )
    column_filter = DropColumns(["a", "b", "c"],["d"], input)
    output = column_filter.filter_columns(["a", "b", "c"])
    blocks = column_filter.blocks()
    assert set(["a", "b"]) == set(output)
    assert set(["a", "b", "dv"]) == set(blocks.keys())
    assert [ "d" ] == blocks["dv"]
    assert [ "a" ] == blocks["a"]
    assert set(["b1", "b2"]) == set(blocks["b"])

def test_should_detect_null_columns():
    input = pd.DataFrame(
        {"a" : [np.NaN, np.NaN, np.NaN],
         "b1": [np.NaN, np.NaN, np.NaN],
         "b2": [1, 2, np.NaN],
         "e": [1, 2, 3],
         "d": [1, 2, 3]}
    )
    column_filter = DropColumns(["a", "b", "c"],["d"], input)
    output = column_filter.filter_columns(["a", "b", "c"])
    assert set(["b"]) == set(output);
