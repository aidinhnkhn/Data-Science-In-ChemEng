# Methanol Synthesis Loop: Surrogate Modeling and Optimization

This project builds a machine-learning surrogate for a methanol synthesis loop
and then uses it for process optimization. The work is split into four tasks:

- **Task 1.** Exploratory data analysis of the process dataset.
- **Task 2.** Two baselines: a linear model (Elastic Net) and a nonlinear model
  (XGBoost).
- **Task 3.** Four neural-network architectures (Deep MLP, Residual MLP,
  Multi-Head MLP, Physics-Informed NN) and a final constrained deep ensemble.
- **Task 4.** Constrained optimization that uses the best surrogate to maximize
  methanol recovery.

The data has 9 inputs and 19 raw outputs. Four outputs are deterministic or
redundant, so the models predict the remaining 15 outputs.

This README explains how to reproduce every result from scratch.

## Repository layout

| Path | What it is |
|---|---|
| `dataset/` | Training and test CSV files plus the input and output schema files. |
| `EDA&ElasticNet.ipynb` | Task 1 and Task 2(i). |
| `XGBoost.ipynb` | Task 2(ii). |
| `NN_DeepMLP.ipynb`, `NN_ResidualMLP.ipynb`, `NN_MultiHead.ipynb`, `NN_PINN.ipynb` | The four Task 3 architectures. |
| `NN_Ensemble.ipynb` | The final surrogate (constrained deep ensemble). |
| `Task4_Optimization.ipynb` | Task 4 constrained optimization. |
| `nn_utils.py` | Shared code for every neural-network notebook: data pipeline, scaling, train/validation split, training loop, metrics, and the physics constraint layers. |
| `models/` | Pre-trained checkpoints, grouped by task. |
| `report/` | LaTeX report source and figures. |

## Environment setup

The project uses Python 3.12. Create a virtual environment and install the
dependencies:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is optional. The models are small and run fine on CPU. I trained them on CPU, So maybe on GPU you will get slightly different results due to floating point differences.

## Data

The data is in the repository, so there is nothing to download from an external source.

| File | Content |
|---|---|
| `dataset/train.csv` | Training set, 8000 rows. |
| `dataset/test.csv` | Test set, 2000 rows. |
| `dataset/inputs_schema.md`, `dataset/outputs_schema.md` | Variable names, units, and ranges. |

The test set is held out. It is used only for the final evaluation, never for
tuning. The validation set is carved out of the training set inside the
notebooks.

## How to reproduce (run order)

Run the notebooks from the repository root, in this order. Every neural-network
notebook imports `nn_utils.py`, so the working directory must be the repo root.

| Step | Notebook | Task | Writes |
|---|---|---|---|
| 1 | `EDA&ElasticNet.ipynb` | Task 1 EDA and Task 2(i) Elastic Net | EDA and parity figures |
| 2 | `XGBoost.ipynb` | Task 2(ii) XGBoost | `models/task2/xgboost.joblib` |
| 3 | `NN_DeepMLP.ipynb` | Task 3 architecture 1 | `models/task3/deep.pt` |
| 4 | `NN_ResidualMLP.ipynb` | Task 3 architecture 2 | `models/task3/residual.pt` |
| 5 | `NN_MultiHead.ipynb` | Task 3 architecture 3 | `models/task3/multihead.pt` |
| 6 | `NN_PINN.ipynb` | Task 3 architecture 4 (KKT-hPINN) | `models/task3/pinn.pt` |
| 7 | `NN_Ensemble.ipynb` | Final surrogate (constrained deep ensemble) | `models/task3/ensemble_pinn.pt`, `models/task3_members/*.pt` |
| 8 | `Task4_Optimization.ipynb` | Task 4 constrained optimization | `models/task4/optima.json`, `pareto.csv`, `optima_table.csv` |

Notes:

- The neural-network notebooks set `RANDOM_STATE = 42`, so the splits and
  training are repeatable. Changing This value will result in different results for certain.
- Run the steps in order. Step 7 reuses the Residual MLP configuration as one of
  its ensemble members, and step 8 loads `ensemble_pinn.pt` as its surrogate.
- The trained checkpoints are already included in `models/`. You can skip
  training and just run the evaluation cells, or delete a checkpoint and re-run
  its notebook to train it again from scratch.

## Results (final held-out test set, 2000 rows)

These are the final test scores, measured once on the held-out test set after
selection on validation.

| Model | Test R2 |
|---|---|
| Constrained Deep Ensemble (final surrogate) | 0.998 |
| PINN (KKT-hPINN) | 0.996 |
| Residual MLP | 0.992 |
| XGBoost | 0.987 |
| Multi-Head MLP | 0.984 |
| Deep MLP | 0.983 |
| Elastic Net | 0.938 |

The Constrained Deep Ensemble is the recommended surrogate and the one used for
Task 4. It also satisfies the vapor and liquid mole-fraction closure to about
1e-7. In Task 4 this surrogate maximizes methanol recovery under a heat-duty
limit and a crude-purity floor, and the optimum is reported at two purity floors
taken from the literature.

## Computational cost of a single prediction

Every network here is a stack of fully connected layers, so the cost of one
forward pass is a sum of matrix-vector products. With `s_i = 9` inputs,
`s_o = 15` modeled outputs, hidden width `w`, depth `n`, residual bottleneck
expansion `m = 2`, head count `H = 3` with head width `w_h`, and `M = 5`
ensemble members, the orders and the measured times per single entry are:

| Model | Order of one forward pass | Parameters | Latency, one entry | Per entry in a large batch |
|---|---|---:|---:|---:|
| Deep MLP (w=128, n=4) | O(s_i w + n w^2 + w s_o) | 53,775 | 0.40 ms | 0.55 us |
| Residual MLP (w=64, n=4) | O(s_i w + 2m n w^2 + w s_o) | 68,559 | 0.51 ms | 0.74 us |
| Multi-Head MLP (trunk 64, heads 32) | O(s_i w + n w^2 + H w w_h + w_h s_o) | 7,503 | 0.41 ms | 0.33 us |
| PINN, KKT-hPINN (w=32, n=4) | O(s_i w + 2m n w^2 + w s_o + s_o) | 17,903 | 0.73 ms | 0.78 us |
| Constrained Deep Ensemble (5 x w=32) | O(M (s_i w + 2m n w^2 + w s_o) + s_o) | 89,515 | 2.42 ms | 3.02 us |

Timings are on CPU (PyTorch 2.5.1, 14 threads), median over repeated runs.
Two things are worth knowing when reading the table:

- At batch size 1 the time is dominated by the fixed per-layer framework
  overhead, not by the arithmetic. That is why the 7.5k-parameter Multi-Head
  model is not faster than the 54k-parameter Deep MLP for one entry, and why
  the ensemble costs five times the single PINN: it runs five backbones one
  after another. The last column, where the cost is averaged over a batch of
  4096 entries, reflects the real arithmetic and follows the orders above.
- The hard constraint layers (vapor projection, liquid closure) are O(s_o)
  with no trainable weights, so enforcing the physics exactly adds no
  meaningful cost at prediction time.

In every case the dominant term is n w^2 (times M for the ensemble), and one
prediction stays under about 2.5 ms even in the worst case, so the surrogate
is orders of magnitude cheaper than the process simulator it replaces.
