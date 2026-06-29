# Input variables

Each row of `train.csv` and `test.csv` corresponds to one steady-state operating point.
The 9 columns below define the operating conditions of the process.

| Column | Units | Range | Physical meaning |
|---|---|---|---|
| `n1_total`     | kmol/h | [800, 1600]   | Total molar flow of **Feed 1** (syngas stream). |
| `y_CO_1`       | —      | [0.25, 0.45]  | CO mole fraction in Feed 1. |
| `y_H2O_1`      | —      | [0.00, 0.04]  | Water mole fraction in Feed 1 (syngas impurity). The remaining fraction of Feed 1 is H₂: `y_H2_1 = 1 − y_CO_1 − y_H2O_1`. |
| `n2_total`     | kmol/h | [0, 400]      | Total molar flow of **Feed 2** (pure H₂ make-up). May be zero. |
| `T1`           | K      | [290, 320]    | Temperature of Feed 1 at the process boundary (before HX1). |
| `T_reactor_in` | K      | [470, 540]    | Reactor inlet temperature, after HX1 preheat and mixing with Feed 2. |
| `P_R`          | bar    | [50, 80]      | Reactor (and flash) pressure. No pressure letdown between reactor and flash. |
| `T_flash`      | K      | [290, 360]    | Flash separator temperature. The cooling between reactor outlet and flash is not an exchanger of interest in this project. |
| `T4`           | K      | [280, 340]    | Temperature of stream 4 (liquid product) after HX2. Constrained to be ≤ `T_flash − 2` K (HX2 can only cool). |

### Notes

- Feed 2 is assumed to enter at `T2 = 298.15` K (fixed, not an input).
- All temperatures are absolute (Kelvin).
- All heat duties are derived quantities — not inputs.
- Molar flows are on a wet basis (water in Feed 1 is included in `n1_total`).
