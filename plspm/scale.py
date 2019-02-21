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

class _Numeric:

    def blah(self):
        print("wow!")

class _Raw:

    def blah(self):
        print("wow!")


class _Ordinal:

    def blah(self):
        print("wow!")


class _Nominal:

    def blah(self):
        print("wow!")


NUM = _Numeric()
RAW = _Raw()
ORD = _Ordinal()
NOM = _Nominal()
