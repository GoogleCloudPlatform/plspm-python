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

import re

class DropColumns:
    """Drops columns that are not in the lists of dependent and independent variables provided"""

    __variable_regex = re.compile("([^\d]*)\d+")

    def __init__(self, iv, dv, mv):
        self.__filtered = None
        self.__present = None
        self.__blocks = {}
        present = []
        columns_to_drop = []
        self.__blocks["dv"] = []
        for column in list(mv):
            if not mv[column].isnull().all():
                if column in iv:
                    present.append(column)
                    self.__blocks[column] = [ column ]
                    continue
                if column in dv:
                    self.__blocks["dv"].append(column)
                    continue
                re_result = DropColumns.__variable_regex.match(column)
                if re_result and re_result.group(1) in iv:
                    iv_name = re_result.group(1)
                    present.append(iv_name)
                    if iv_name not in self.__blocks:
                        self.__blocks[iv_name] = []
                    self.__blocks[iv_name].append(column)
                    continue
            columns_to_drop.append(column)
        self.__present = set(present)
        self.__filtered = mv.drop(columns_to_drop, axis=1)

    def filter(self):
        return self.__filtered

    def filter_columns(self, columns):
        return list(self.__present & set(columns))

    def blocks(self):
        return self.__blocks
