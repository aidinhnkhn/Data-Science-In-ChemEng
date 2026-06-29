# Project dataset

## Contents

| File | Purpose |
|---|---|
| `train.csv`         | Training set, 8 000 rows. Use it for exploratory analysis and to fit your models. |
| `test.csv`          | Held-out test set, 2 000 rows. Reserve it for the final assessment of your trained surrogate. Do **not** use it for hyperparameter tuning. |
| `inputs_schema.md`  | Names, units, ranges and physical meaning of the input variables. |
| `outputs_schema.md` | Names, units and physical meaning of the output variables. |

Each CSV has the same 28 columns: 9 inputs followed by 19 outputs. The training
and test sets cover the same input domain but were drawn independently of each
other; the test set is not a random subset of the training set. You are
expected to carve a validation split out of the training set yourself.

## Conventions

- Temperatures in **Kelvin**.
- Pressures in **bar** (absolute).
- Molar flows in **kmol/h**.
- Heat-exchanger duties in **MW**, with `Q > 0` for heating and `Q < 0` for cooling.
- Mole fractions are dimensionless and sum to 1 within each phase that exists.
