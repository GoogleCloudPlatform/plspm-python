# plspm

_Please note: This is not an officially supported Google product._

**plspm** is a Python package dedicated to Partial Least Squares Path Modeling (PLS-PM) analysis. It is a partial port of the R package [plspm](https://github.com/gastonstat/plspm).

Currently it will calculate modes A (for reflective relationships) and B (for formative relationships) with non-metric numerical data using centroid, factorial, and path schemes.

## Installation

## Example

Example with the classic Russett data (original data set)

```
#!/usr/bin/python3
import pandas as pd, plspm.scheme as scheme, plspm.plspm as plspm, plspm.mode as mode, plspm.config as c

russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
lvs = ["AGRI", "IND", "POLINS"]
rus_path = pd.DataFrame(
    [[0, 0, 0],
     [0, 0, 0],
     [1, 1, 0]],
    index=lvs,
    columns=lvs)
config = c.Config(rus_path)
config.add_lv("AGRI", mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
config.add_lv("IND", mode.A, c.MV("gnpr"), c.MV("labo"))
config.add_lv("POLINS", mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))

plspm_calc = plspm.Plspm(russa, config, scheme.CENTROID, 100, 0.0000001)

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

## Maintainers

[Jez Humble](https://continuousdelivery.com/)
  (`humble at google.com`)
  
[Nicole Forsgren](https://nicolefv.com/)
  (`nicolefv at google.com`)
