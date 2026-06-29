# Output variables

The CSVs contain 19 output columns following the 9 input columns.

## Heat-exchanger duties

| Column | Units | Sign convention | Meaning |
|---|---|---|---|
| `Q1_MW` | MW | positive = heating | Heat duty of HX1 (preheater on Feed 1). |
| `Q2_MW` | MW | negative = cooling | Heat duty of HX2 (cooler on the flash bottom). |

## Reactor outputs

| Column | Units | Meaning |
|---|---|---|
| `T_reactor_out` | K | Adiabatic reactor outlet temperature. Strongly depends on `T_reactor_in`, `P_R`, and feed composition. |
| `X_CO`          | — | CO conversion: `(CO_in − CO_out) / CO_in` over the reactor. |

## Stream 3 (vapor from flash top)

| Column | Units | Meaning |
|---|---|---|
| `T3`        | K       | Stream 3 temperature = `T_flash`. |
| `P3`        | bar     | Stream 3 pressure = `P_R`. |
| `n3_total`  | kmol/h  | Total molar flow of stream 3. |
| `y_CO_3`    | —       | CO mole fraction in stream 3. |
| `y_H2_3`    | —       | H₂ mole fraction in stream 3. |
| `y_MeOH_3`  | —       | Methanol mole fraction in stream 3. |
| `y_H2O_3`   | —       | H₂O mole fraction in stream 3. |

## Stream 4 (liquid product after HX2)

Stream-4 temperature is the input `T4`; no separate output column is written for it. The other state variables are:

| Column | Units | Meaning |
|---|---|---|
| `P4`        | bar     | Stream 4 pressure = `P_R`. |
| `n4_total`  | kmol/h  | Total molar flow of stream 4. Can be 0 in the degenerate case where the flash does not produce any liquid. |
| `x_CO_4`    | —       | CO mole fraction in stream 4. |
| `x_H2_4`    | —       | H₂ mole fraction in stream 4. |
| `x_MeOH_4`  | —       | Methanol mole fraction in stream 4. |
| `x_H2O_4`   | —       | H₂O mole fraction in stream 4. |

## KPIs

| Column | Units | Meaning |
|---|---|---|
| `MeOH_recovery`       | — | Fraction of the methanol produced in the reactor that leaves the process in stream 4, i.e. `n_MeOH_stream4 / n_MeOH_produced`. Zero when the flash does not produce any liquid. |
| `MeOH_purity_stream4` | — | Methanol mole fraction in stream 4, i.e. `x_MeOH_4`. Zero when the flash does not produce any liquid. |

### Notes

- Element conservation (C, H, O) holds to better than 1×10⁻⁸ relative error on every row.
- `y_CO_3 + y_H2_3 + y_MeOH_3 + y_H2O_3 = 1` exactly.
- `x_CO_4 + x_H2_4 + x_MeOH_4 + x_H2O_4 = 1` exactly on rows for which a liquid phase exists; all four mole fractions are zero on rows for which `n4_total = 0`.
- The sum-to-one constraints introduce one redundant degree of freedom per stream. You may choose to predict only three of the four mole fractions and recover the fourth by difference — this is a design choice to discuss in your report.
