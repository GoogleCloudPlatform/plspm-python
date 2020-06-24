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

import plspm.config as c, pandas as pd, numpy as np, plspm.inner_model as im, plspm.outer_model as om
from plspm.weights import WeightsCalculatorFactory


def _create_summary(data: pd.DataFrame, original):
    summary = pd.DataFrame(0, index=data.columns, columns=["original", "mean", "std.error", "perc.025", "perc.975", "t stat."])
    summary.loc[:, "mean"] = data.mean(axis=0)
    summary.loc[:, "std.error"] = data.std(axis=0)
    summary.loc[:, "perc.025"] = data.quantile(0.025, axis=0)
    summary.loc[:, "perc.975"] = data.quantile(0.975, axis=0)
    summary.loc[:, "original"] = original
    summary.loc[:, "t stat."] = original / data.std(axis=0)
    return summary


class Bootstrap:
    """Performs bootstrap validation to determine the statistical significance of the model.

    Setting ``bootstrap=True`` when constructing :class:`.Plspm` will perform bootstrap validation. Calling :meth:`~.Plspm.bootstrap` on :class:`.Plspm` will return an instance of this class, from which the bootstrapping results can be retrieved by calling the methods listed below.
    """
    def __init__(self, config: c.Config, data: pd.DataFrame, inner_model: im.InnerModel, outer_model: om.OuterModel,
                 calculator: WeightsCalculatorFactory, iterations: int):
        observations = data.shape[0]
        weights = pd.DataFrame(columns=data.columns)
        r_squared = pd.DataFrame(columns=inner_model.r_squared().index)
        total_effects = pd.DataFrame(columns=inner_model.effects().index)
        paths = pd.DataFrame(columns=inner_model.effects().index)
        loadings = pd.DataFrame(columns=data.columns)
        for i in range(1, iterations):
            try:
                boot_observations = np.random.randint(observations, size=observations)
                boot_data = config.treat(data.iloc[boot_observations, :])
                _final_data, _scores, _weights = calculator.calculate(boot_data)
                weights = weights.append(_weights.T, ignore_index=True)
                inner_model = im.InnerModel(config.path(), _scores)
                r_squared = r_squared.append(inner_model.r_squared().T, ignore_index=True)
                total_effects = total_effects.append(inner_model.effects().loc[:, "total"].T, ignore_index=True)
                paths = paths.append(inner_model.effects().loc[:, "direct"].T, ignore_index=True)
                loadings = loadings.append(
                    (_scores.apply(lambda s: _final_data.corrwith(s)) * config.odm()).sum(axis=1), ignore_index=True)
            except:
                pass
        self.__weights = _create_summary(weights, outer_model.model().loc[:, "weight"])
        self.__r_squared = _create_summary(r_squared, inner_model.r_squared()).loc[inner_model.endogenous(), :]
        self.__total_effects = _create_summary(total_effects, inner_model.effects().loc[:, "total"])
        self.__paths = _create_summary(paths, inner_model.effects().loc[:, "direct"])
        self.__loading = _create_summary(loadings, outer_model.model().loc[:, "loading"])

    def weights(self) -> pd.DataFrame:
        """Outer weights calculated from bootstrap validation."""
        return self.__weights

    def r_squared(self) -> pd.DataFrame:
        """R squared for latent variables calculated from bootstrap validation."""
        return self.__r_squared

    def total_effects(self) -> pd.DataFrame:
        """Total effects for paths calculated from bootstrap validation."""
        return self.__total_effects

    def paths(self) -> pd.DataFrame:
        """Direct effects for paths calculated from bootstrap validation."""
        return self.__paths[self.__paths["mean"] != 0]

    def loading(self) -> pd.DataFrame:
        """Loadings of manifest variables calculated from bootstrap validation."""
        return self.__loading
