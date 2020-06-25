# PLSPM: A library implementing Partial Least Squares Path Modeling [![PyPI version](https://badge.fury.io/py/plspm.svg)](https://badge.fury.io/py/plspm)

_Please note: This is not an officially supported Google product._

**plspm** is a Python 3 package dedicated to Partial Least Squares Path Modeling (PLS-PM) analysis. It is a port of the R package [plspm](https://github.com/gastonstat/plspm).

PLSPM (partial least squares path modeling) is a correlation-based structural equation modeling (SEM) algorithm. It allows for estimation of complex cause-effect or prediction models using latent/manifest variables. 

PLSPM may be preferred to other SEM methods for several reasons: it is a method that is appropriate for exploratory research, can be used with small-to-medium sample sizes (as well as large data sets), and does not require assumptions of multivariate normality. (See Hulland, J. (1999). Use of partial least squares (PLS) in strategic management research: a review of four recent studies. Strategic management journal, 20(2), 195-204.) In contrast to covariance-based SEM (CBSEM), goodness of fit is less important, because the purpose of the algorithm is to optimize for prediction of the dependent variable vs. fit of data to a predetermined model. (See "goodness of fit" vs "goodness of model" in Chin, W. W. (2010). How to write up and report PLS analyses. In Handbook of partial least squares (pp. 655-690). Springer, Berlin, Heidelberg.)

This library will calculate using modes A (for reflective relationships) and B (for formative relationships) with metric and non-metric numerical data using centroid, factorial, and path schemes. Bootstrap validation is available, and reliability measures are also calculated using the same methods as the original R library.

## Installation
You can install the latest version of this package using pip:

```sh
python3 -m pip install --user plspm
```

It's hosted on pypi: https://pypi.org/project/plspm/

## Use

plspm expects to get a Pandas DataFrame containing your data. You start by creating a [Config](https://plspm.readthedocs.io/en/latest/source/plspm.config.html) object with the details of the model, and then pass it, along with the data and optionally some further configuration, to an instance of [Plspm](https://plspm.readthedocs.io/en/latest/source/plspm.plspm.html). Use the examples below to get started, or browse the [documentation](https://plspm.readthedocs.io/) (start with [Config](https://plspm.readthedocs.io/en/latest/source/plspm.config.html) and [Plspm](https://plspm.readthedocs.io/en/latest/source/plspm.plspm.html))

## Examples

### PLS-PM with metric data

Typical example with a Customer Satisfaction Model

```py
#!/usr/bin/env python3
import pandas as pd, plspm.config as c
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode

satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)

structure = c.Structure()
structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
structure.add_path(["QUAL"], ["VAL", "SAT"])
structure.add_path(["VAL"], ["SAT"])
structure.add_path(["SAT"], ["LOY"])

config = c.Config(structure.path(), scaled=False)
config.add_lv_with_columns_named("IMAG", Mode.A, satisfaction, "imag")
config.add_lv_with_columns_named("EXPE", Mode.A, satisfaction, "expe")
config.add_lv_with_columns_named("QUAL", Mode.A, satisfaction, "qual")
config.add_lv_with_columns_named("VAL", Mode.A, satisfaction, "val")
config.add_lv_with_columns_named("SAT", Mode.A, satisfaction, "sat")
config.add_lv_with_columns_named("LOY", Mode.A, satisfaction, "loy")

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

### PLS-PM with Nonmetric Data

Example with the classic Russett data (original data set)

```py
#!/usr/bin/env python3
import pandas as pd, plspm.config as c
from plspm.plspm import Plspm
from plspm.scale import Scale
from plspm.scheme import Scheme
from plspm.mode import Mode

russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)

structure = c.Structure()
structure.add_path(["AGRI", "IND"], ["POLINS"])

config = c.Config(structure.path(), default_scale=Scale.NUM)
config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))

plspm_calc = Plspm(russa, config, Scheme.CENTROID, 100, 0.0000001)
print(plspm_calc.inner_summary())
print(plspm_calc.effects())
```

This will produce the output:
```
              type  r_squared  block_communality  mean_redundancy       ave
AGRI     Exogenous   0.000000           0.739560         0.000000  0.739560
IND      Exogenous   0.000000           0.907524         0.000000  0.907524
POLINS  Endogenous   0.592258           0.565175         0.334729  0.565175

   from      to    direct  indirect     total
0  AGRI  POLINS  0.225639       0.0  0.225639
1   IND  POLINS  0.671457       0.0  0.671457
```

#### Example 2: Different Scaling

PLS-PM using data set `russa`, and different scaling

```py
#!/usr/bin/python3
import pandas as pd, plspm.config as c, plspm.util as util
from plspm.plspm import Plspm
from plspm.scale import Scale
from plspm.scheme import Scheme
from plspm.mode import Mode

russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)

structure = c.Structure()
structure.add_path(["AGRI", "IND"], ["POLINS"])
config = c.Config(structure.path(), default_scale=Scale.NUM)
config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
config.add_lv("IND", Mode.A, c.MV("gnpr", Scale.ORD), c.MV("labo", Scale.ORD))
config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo", Scale.NOM), c.MV("inst"))

plspm_calc = Plspm(russa, config, Scheme.CENTROID, 100, 0.0000001)
```

#### Example 3: Missing Data

```py
#!/usr/bin/env python3
import pandas as pd, plspm.config as c
from plspm.plspm import Plspm
from plspm.scale import Scale
from plspm.scheme import Scheme
from plspm.mode import Mode

russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
russa.iloc[0, 0] = np.NaN
russa.iloc[3, 3] = np.NaN
russa.iloc[5, 5] = np.NaN

structure = c.Structure()
structure.add_path(["AGRI", "IND"], ["POLINS"])
config = c.Config(structure.path(), default_scale=Scale.NUM)
config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))

plspm_calc = Plspm(russa, config, Scheme.CENTROID, 100, 0.0000001)
```

## Maintainers

[Jez Humble](https://continuousdelivery.com/)
  (`humble at google.com`)

[Nicole Forsgren](https://nicolefv.com/)
  (`nicolefv at google.com`)
