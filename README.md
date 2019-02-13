# plspm

_Please note: This is not an officially supported Google product._

**plspm** is a Python package dedicated to Partial Least Squares Path Modeling (PLS-PM) analysis. It is a partial port of the R package [plspm](https://github.com/gastonstat/plspm). Currently it will calculate Mode A (for reflective models) with non-metric numerical data using centroid, factorial, and path schemes.

## Installation

## Example

Example with the classic Russett data (original data set)

```
#!/usr/bin/python3
import pandas as pd, plspm.scheme as scheme, plspm.plspm as plspm

russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
rus_path = pd.DataFrame(
    [[0, 0, 0],
     [0, 0, 0],
     [1, 1, 0]],
    index=["AGRI", "IND", "POLINS"],
    columns=["AGRI", "IND", "POLINS"])
rus_blocks = {"AGRI": ["gini", "farm", "rent"],
              "IND": ["gnpr", "labo"],
              "POLINS": ["ecks", "death", "demo", "inst"]}

plspm_calc = plspm.Plspm(russa, rus_path, rus_blocks, scheme.CENTROID, 100, 0.0000001)
```

## Maintainers

[Jez Humble](https://continuousdelivery.com/)
  (`humble at google.com`)
  
[Nicole Forsgren](https://nicolefv.com/)
  (`nicolefv at google.com`)
