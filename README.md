# plspm

_Please note: This is not an officially supported Google product._

**plspm** is a Python 3 package dedicated to Partial Least Squares Path Modeling (PLS-PM) analysis. It is a partial port of the R package [plspm](https://github.com/gastonstat/plspm).

Currently it will calculate modes A (for reflective relationships) and B (for formative relationships) with metric and non-metric numerical data using centroid, factorial, and path schemes. At present the library does not yet calculate unidimensionality, nor will it perform bootstrapping. Missing values in  non-metric data are also not handled.

## Installation
s
You can install the latest version of this package using pip:

`python3 -m pip install --user plspm`

It's hosted on pypi: https://pypi.org/project/plspm/

## Examples

### PLS-PM with metric data

Typical example with a Customer Satisfaction Model

```
#!/usr/bin/python3
import pandas as pd, plspm.util as util, plspm.config as c
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode

satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
lvs = ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]
sat_path_matrix = pd.DataFrame(
    [[0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 1, 1, 0, 0, 0],
     [1, 1, 1, 1, 0, 0],
     [1, 0, 0, 0, 1, 0]],
    index=lvs, columns=lvs)
config = c.Config(sat_path_matrix, scaled=False)
config.add_lv_with_columns_named("imag", satisfaction, "IMAG", Mode.A)
config.add_lv_with_columns_named("expe", satisfaction, "EXPE", Mode.A)
config.add_lv_with_columns_named("qual", satisfaction, "QUAL", Mode.A)
config.add_lv_with_columns_named("val", satisfaction, "VAL", Mode.A)
config.add_lv_with_columns_named("sat", satisfaction, "SAT", Mode.A)
config.add_lv_with_columns_named("loy", satisfaction, "LOY", Mode.A)
plspm_calc = Plspm(satisfaction, config, Scheme.CENTROID)
print(plspm_calc.inner_summary())
print(plspm_calc.path_coefficients())
```

This will produce the output:
```
            type  r_squared  block_communality  mean_redundancy       ave
EXPE  Endogenous   0.335194           0.616420         0.206620  0.616420
IMAG   Exogenous   0.000000           0.582269         0.000000  0.582269
LOY   Endogenous   0.509923           0.639052         0.325867  0.639052
QUAL  Endogenous   0.719688           0.658572         0.473966  0.658572
SAT   Endogenous   0.707321           0.758891         0.536779  0.758891
VAL   Endogenous   0.590084           0.664416         0.392061  0.664416

          IMAG      EXPE      QUAL       VAL       SAT  LOY
IMAG  0.000000  0.000000  0.000000  0.000000  0.000000    0
EXPE  0.578959  0.000000  0.000000  0.000000  0.000000    0
QUAL  0.000000  0.848344  0.000000  0.000000  0.000000    0
VAL   0.000000  0.105478  0.676656  0.000000  0.000000    0
SAT   0.200724 -0.002754  0.122145  0.589331  0.000000    0
LOY   0.275150  0.000000  0.000000  0.000000  0.495479    0
```

### PLS-PM with nonmetric data

Example with the classic Russett data (original data set)

```
#!/usr/bin/python3
import pandas as pd, plspm.config as c
from plspm.plspm import Plspm
from plspm.scale import Scale
from plspm.scheme import Scheme
from plspm.mode import Mode

russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
lvs = ["AGRI", "IND", "POLINS"]
rus_path = pd.DataFrame(
    [[0, 0, 0],
     [0, 0, 0],
     [1, 1, 0]],
    index=lvs,
    columns=lvs)
config = c.Config(rus_path, default_scale=Scale.NUM)
config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))

plspm_calc = Plspm(russa, config, Scheme.CENTROID, 100, 0.0000001)

print(plspm_calc.inner_summary())
print(plspm_calc.path_coefficients())
```

This will produce the output:
```
              type  r_squared  block_communality  mean_redundancy       ave
AGRI     Exogenous   0.000000           0.739560         0.000000  0.739560
IND      Exogenous   0.000000           0.907524         0.000000  0.907524
POLINS  Endogenous   0.592258           0.565175         0.334729  0.565175

            AGRI       IND  POLINS
AGRI    0.000000  0.000000       0
IND     0.000000  0.000000       0
POLINS  0.225639  0.671457       0
```

#### Example 2

PLS-PM using data set `russa`, and different scaling

```
#!/usr/bin/python3
import pandas as pd, plspm.config as c, plspm.util as util
from plspm.plspm import Plspm
from plspm.scale import Scale
from plspm.scheme import Scheme
from plspm.mode import Mode

def russa_path_matrix():
    lvs = ["AGRI", "IND", "POLINS"]
    return pd.DataFrame(
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 0]],
        index=lvs, columns=lvs)

russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
config = c.Config(russa_path_matrix(), default_scale=Scale.NUM)
config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
config.add_lv("IND", Mode.A, c.MV("gnpr", Scale.ORD), c.MV("labo", Scale.ORD))
config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo", Scale.NOM), c.MV("inst"))

plspm_calc = Plspm(russa, config, Scheme.CENTROID, 100, 0.0000001)
```

## Maintainers

[Jez Humble](https://continuousdelivery.com/)
  (`humble at google.com`)
  
[Nicole Forsgren](https://nicolefv.com/)
  (`nicolefv at google.com`)
