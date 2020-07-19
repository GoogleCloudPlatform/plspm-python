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
import threading
from plspm.weights import WeightsCalculatorFactory
from plspm.estimator import Estimator

def _create_summary(data: pd.DataFrame, original):
    summary = pd.DataFrame(0, index=data.columns, columns=["original", "mean", "std.error", "perc.025", "perc.975", "t stat."])
    summary.loc[:, "mean"] = data.mean(axis=0)
    summary.loc[:, "std.error"] = data.std(axis=0)
    summary.loc[:, "perc.025"] = data.quantile(0.025, axis=0)
    summary.loc[:, "perc.975"] = data.quantile(0.975, axis=0)
    summary.loc[:, "original"] = original
    summary.loc[:, "t stat."] = original / data.std(axis=0)
    return summary


class BootstrapThread (threading.Thread):
    def __init__(self, config: c.Config, data: pd.DataFrame, inner_model: im.InnerModel, calculator: WeightsCalculatorFactory, iterations: int):
        threading.Thread.__init__(self)
        self.__config = config
        self.__data = data
        self.__calculator = calculator
        self.__iterations = iterations
        self.__weights = pd.DataFrame(columns=data.columns)
        self.__r_squared = pd.DataFrame(columns=inner_model.r_squared().index)
        self.__total_effects = pd.DataFrame(columns=inner_model.effects().index)
        self.__paths = pd.DataFrame(columns=inner_model.effects().index)
        self.__loadings = pd.DataFrame(columns=data.columns)

    def run(self):
        observations = self.__data.shape[0]
        estimator = Estimator(self.__config)
        for i in range(0, self.__iterations):
            try:
                boot_observations = np.random.randint(observations, size=observations)
                _final_data, _scores, _weights = estimator.estimate(self.__calculator, self.__data.iloc[boot_observations, :])
                self.__weights = self.__weights.append(_weights.T, ignore_index=True)
                inner_model = im.InnerModel(self.__config.path(), _scores)
                self.__r_squared = self.__r_squared.append(inner_model.r_squared().T, ignore_index=True)
                self.__total_effects = self.__total_effects.append(inner_model.effects().loc[:, "total"].T, ignore_index=True)
                self.__paths = self.__paths.append(inner_model.effects().loc[:, "direct"].T, ignore_index=True)
                self.__loadings = self.__loadings.append(
                    (_scores.apply(lambda s: _final_data.corrwith(s)) * self.__config.odm(self.__config.path())).sum(axis=1), ignore_index=True)
            except:
                pass

    def weights(self):
        return self.__weights

    def r_squared(self):
        return self.__r_squared

    def total_effects(self):
        return self.__total_effects

    def paths(self):
        return self.__paths

    def loadings(self):
        return self.__loadings


class Bootstrap:
    """Performs bootstrap validation to determine the statistical significance of the model.

    Setting ``bootstrap=True`` when constructing :class:`.Plspm` will perform bootstrap validation. Calling :meth:`~.Plspm.bootstrap` on :class:`.Plspm` will return an instance of this class, from which the bootstrapping results can be retrieved by calling the methods listed below.
    """
    def __init__(self, config: c.Config, data: pd.DataFrame, inner_model: im.InnerModel, outer_model: om.OuterModel,
                 calculator: WeightsCalculatorFactory, iterations: int, num_threads: int):
        weights = pd.DataFrame(columns=data.columns)
        r_squared = pd.DataFrame(columns=inner_model.r_squared().index)
        total_effects = pd.DataFrame(columns=inner_model.effects().index)
        paths = pd.DataFrame(columns=inner_model.effects().index)
        loadings = pd.DataFrame(columns=data.columns)

        threads = []
        for t in range(0, num_threads):
            thread = BootstrapThread(config, data, inner_model, calculator, iterations // num_threads)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
            weights = weights.append(thread.weights())
            r_squared = r_squared.append(thread.r_squared())
            total_effects = total_effects.append(thread.total_effects())
            paths = paths.append(thread.paths())
            loadings = loadings.append(thread.loadings())

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
