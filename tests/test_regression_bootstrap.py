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


def test_bootstrap_metric():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    config = c.Config(satisfaction_path_matrix(), scaled=False)
    config.add_lv_with_columns_named("IMAG", Mode.A, satisfaction, "imag")
    config.add_lv_with_columns_named("EXPE", Mode.A, satisfaction, "expe")
    config.add_lv_with_columns_named("QUAL", Mode.A, satisfaction, "qual")
    config.add_lv_with_columns_named("VAL", Mode.A, satisfaction, "val")
    config.add_lv_with_columns_named("SAT", Mode.A, satisfaction, "sat")
    config.add_lv_with_columns_named("LOY", Mode.A, satisfaction, "loy")

    plspm_calc = Plspm(satisfaction, config, bootstrap=True)
    expected_boot_weights = pd.read_csv("file:tests/data/satisfaction_boot_weights.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_boot_weights),
                        util.sort_cols(plspm_calc.bootstrap().weights()), rtol=0.5)

    expected_boot_rsquared = pd.read_csv("file:tests/data/satisfaction_boot_rsquared.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_boot_rsquared),
                        util.sort_cols(plspm_calc.bootstrap().r_squared()), rtol=0.5)

    expected_boot_total_effects = pd.read_csv("file:tests/data/satisfaction_boot_total_effects.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_boot_total_effects),
                        util.sort_cols(plspm_calc.bootstrap().total_effects()), rtol=0.5)

    expected_boot_paths = pd.read_csv("file:tests/data/satisfaction_boot_paths.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_boot_paths),
                        util.sort_cols(plspm_calc.bootstrap().paths()), rtol=5)

    expected_boot_loadings = pd.read_csv("file:tests/data/satisfaction_boot_loadings.csv", index_col=0)
    npt.assert_allclose(util.sort_cols(expected_boot_loadings),
                        util.sort_cols(plspm_calc.bootstrap().loadings()), rtol=0.5)
