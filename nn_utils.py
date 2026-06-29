"""
Shared utilities for the Task 3 neural-surrogate notebooks.

Every NN_*.ipynb in this project imports from here so the data pipeline,
scaling, train/val split, training loop, and metrics are identical across
all four architectures. The only thing that varies between notebooks is the
model class (and, for the PINN, the loss function).
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants. Keep aligned with EDA&ElasticNet.ipynb and XGBoost.ipynb
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

INPUT_COLS: list[str] = [
    "n1_total", "y_CO_1", "y_H2O_1", "n2_total",
    "T1", "T_reactor_in", "P_R", "T_flash", "T4",
]

# Full 19-output column order in the CSV
OUTPUT_COLS: list[str] = [
    "Q1_MW", "Q2_MW", "T_reactor_out", "X_CO",
    "T3", "P3", "n3_total",
    "y_CO_3", "y_H2_3", "y_MeOH_3", "y_H2O_3",
    "P4", "n4_total",
    "x_CO_4", "x_H2_4", "x_MeOH_4", "x_H2O_4",
    "MeOH_recovery", "MeOH_purity_stream4",
]

# Deterministic / redundant outputs, recovered by direct assignment, not modeled.
DETERMINISTIC_FROM_INPUT: dict[str, str] = {
    "T3": "T_flash", "P3": "P_R", "P4": "P_R",
}
REDUNDANT_FROM_OUTPUT: dict[str, str] = {
    "MeOH_purity_stream4": "x_MeOH_4",
}

# 15 outputs we actually model with the NN.
MODEL_OUTPUT_COLS: list[str] = [
    c for c in OUTPUT_COLS
    if c not in DETERMINISTIC_FROM_INPUT and c not in REDUNDANT_FROM_OUTPUT
]

# Outputs that are trained in log-space (matches XGBoost convention).
LOG_TRANSFORM_OUTPUTS: set[str] = {"x_H2_4"}
LOG_FLOOR = 1e-10  # value used to clip before log() to avoid log(0) for no-liquid rows

# Index lookups used by the PINN constraint loss.
VAPOR_FRACTION_COLS = ["y_CO_3", "y_H2_3", "y_MeOH_3", "y_H2O_3"]
LIQUID_FRACTION_COLS = ["x_CO_4", "x_H2_4", "x_MeOH_4", "x_H2O_4"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    no_liquid_train: pd.Series
    no_liquid_test: pd.Series


def load_data(dataset_dir: str | Path = "dataset") -> DataBundle:
    """Load train/test CSVs and return inputs, full 19-output targets, and the no-liquid masks."""
    dataset_dir = Path(dataset_dir)
    train = pd.read_csv(dataset_dir / "train.csv")
    test = pd.read_csv(dataset_dir / "test.csv")
    return DataBundle(
        X_train=train[INPUT_COLS],
        y_train=train[OUTPUT_COLS],
        X_test=test[INPUT_COLS],
        y_test=test[OUTPUT_COLS],
        no_liquid_train=train["n4_total"].eq(0),
        no_liquid_test=test["n4_total"].eq(0),
    )


# ---------------------------------------------------------------------------
# Scaling pipeline (log on x_H2_4, then StandardScaler on the 15 modeled outputs)
# ---------------------------------------------------------------------------

class TargetTransformer:
    """
    Applies log() to the LOG_TRANSFORM_OUTPUTS columns then StandardScaler to the
    full 15-column target matrix. Fit on train only.

    `transform`         : raw 15-col targets  -> scaled (model space)
    `inverse_transform` : scaled (model space) -> raw 15-col targets
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.log_idx: list[int] = [
            MODEL_OUTPUT_COLS.index(c) for c in LOG_TRANSFORM_OUTPUTS
        ]

    def _apply_log(self, y: np.ndarray) -> np.ndarray:
        y = y.copy()
        for j in self.log_idx:
            y[:, j] = np.log(np.clip(y[:, j], LOG_FLOOR, None))
        return y

    def _invert_log(self, y: np.ndarray) -> np.ndarray:
        y = y.copy()
        for j in self.log_idx:
            y[:, j] = np.exp(y[:, j])
        return y

    def fit(self, y_raw: np.ndarray) -> "TargetTransformer":
        self.scaler.fit(self._apply_log(y_raw))
        return self

    def transform(self, y_raw: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self._apply_log(y_raw))

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        return self._invert_log(self.scaler.inverse_transform(y_scaled))

    def inverse_transform_torch(self, y_scaled: torch.Tensor) -> torch.Tensor:
        """Differentiable inverse-transform for the PINN constraint loss."""
        mean = torch.as_tensor(self.scaler.mean_, dtype=y_scaled.dtype, device=y_scaled.device)
        scale = torch.as_tensor(self.scaler.scale_, dtype=y_scaled.dtype, device=y_scaled.device)
        y_log = y_scaled * scale + mean
        if self.log_idx:
            y_lin = y_log.clone()
            for j in self.log_idx:
                y_lin[:, j] = torch.exp(y_log[:, j])
            return y_lin
        return y_log


@dataclass
class ScaledData:
    X_train_s: np.ndarray
    y_train_s: np.ndarray
    X_val_s: np.ndarray
    y_val_s: np.ndarray
    X_test_s: np.ndarray
    no_liquid_train_fit: np.ndarray  # mask after train/val split (training portion)
    no_liquid_val: np.ndarray
    no_liquid_test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    X_scaler: StandardScaler
    Y_transformer: TargetTransformer


def build_scaled_split(
    data: DataBundle,
    val_size: float = 0.2,
    seed: int = RANDOM_STATE,
) -> ScaledData:
    """
    Carve a stratified train/val split out of the training set (stratified on the
    no-liquid indicator). Fit X_scaler and Y_transformer on the *training portion only*.
    Targets restricted to the 15 MODEL_OUTPUT_COLS.
    """
    y_train_full = data.y_train[MODEL_OUTPUT_COLS].values
    no_liquid_train_full = data.no_liquid_train.astype(int).values
    indices = np.arange(len(data.X_train))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=no_liquid_train_full,
        random_state=seed,
    )

    X_scaler = StandardScaler().fit(data.X_train.values[train_idx])
    Y_transformer = TargetTransformer().fit(y_train_full[train_idx])

    return ScaledData(
        X_train_s=X_scaler.transform(data.X_train.values[train_idx]),
        y_train_s=Y_transformer.transform(y_train_full[train_idx]),
        X_val_s=X_scaler.transform(data.X_train.values[val_idx]),
        y_val_s=Y_transformer.transform(y_train_full[val_idx]),
        X_test_s=X_scaler.transform(data.X_test.values),
        no_liquid_train_fit=no_liquid_train_full[train_idx].astype(bool),
        no_liquid_val=no_liquid_train_full[val_idx].astype(bool),
        no_liquid_test=data.no_liquid_test.values.astype(bool),
        train_idx=train_idx,
        val_idx=val_idx,
        X_scaler=X_scaler,
        Y_transformer=Y_transformer,
    )


def make_dataloaders(
    scaled: ScaledData,
    batch_size: int = 128,
    seed: int = RANDOM_STATE,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) over standardized tensors."""
    gen = torch.Generator().manual_seed(seed)
    train_ds = TensorDataset(
        torch.as_tensor(scaled.X_train_s, dtype=torch.float32),
        torch.as_tensor(scaled.y_train_s, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.as_tensor(scaled.X_val_s, dtype=torch.float32),
        torch.as_tensor(scaled.y_val_s, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    max_epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lr_factor: float = 0.5
    lr_patience: int = 10
    lr_min: float = 1e-6
    early_stopping_patience: int = 20
    grad_clip_norm: float | None = 1.0
    device: str = "cpu"
    log_every: int = 25


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig = TrainConfig(),
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> TrainHistory:
    """
    Generic training loop with AdamW + ReduceLROnPlateau + early stopping +
    best-weights restoration. The model is mutated in place to hold the best
    weights at exit.

    `loss_fn(y_pred, y_true) -> scalar` is the default for the simple
    architectures. The PINN passes a custom callable that adds the constraint
    terms.
    """
    device = torch.device(config.device)
    model.to(device)
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.lr_factor, patience=config.lr_patience, min_lr=config.lr_min,
    )

    history = TrainHistory()
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            bs = xb.size(0)
            running += loss.item() * bs
            n_seen += bs
        train_loss = running / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            v_running = 0.0
            v_seen = 0
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                y_pred = model(xb)
                v_running += loss_fn(y_pred, yb).item() * xb.size(0)
                v_seen += xb.size(0)
            val_loss = v_running / max(v_seen, 1)

        scheduler.step(val_loss)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.lr.append(optimizer.param_groups[0]["lr"])

        if val_loss + 1e-9 < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch == 1 or epoch % config.log_every == 0 or patience == 0:
            print(
                f"  epoch {epoch:3d} | train {train_loss:.5f} | val {val_loss:.5f} "
                f"| lr {optimizer.param_groups[0]['lr']:.2e} | best@{history.best_epoch}"
            )

        if patience >= config.early_stopping_patience:
            print(f"  early stop @ epoch {epoch} (best epoch {history.best_epoch}, "
                  f"val {history.best_val_loss:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ---------------------------------------------------------------------------
# Inference & metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_scaled(model: nn.Module, X_s: np.ndarray, device: str = "cpu",
                   batch_size: int = 1024) -> np.ndarray:
    """Forward `X_s` through the model in batches; return raw model output (still scaled)."""
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    out_chunks = []
    X_t = torch.as_tensor(X_s, dtype=torch.float32, device=dev)
    for start in range(0, X_t.size(0), batch_size):
        out_chunks.append(model(X_t[start:start + batch_size]).cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def predict_raw(
    model: nn.Module,
    X_s: np.ndarray,
    Y_transformer: TargetTransformer,
    device: str = "cpu",
) -> np.ndarray:
    """Predict in raw units (15-column array in MODEL_OUTPUT_COLS order)."""
    return Y_transformer.inverse_transform(predict_scaled(model, X_s, device=device))


def regression_metrics_by_output(
    y_true: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray | pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Per-column RMSE, MAE, R². Mirrors XGBoost.ipynb's helper of the same name."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for j, col in enumerate(columns):
        t = y_true[:, j]
        p = y_pred[:, j]
        r2 = r2_score(t, p) if len(t) >= 2 else np.nan
        rows.append({
            "output": col,
            "RMSE":   float(np.sqrt(mean_squared_error(t, p))),
            "MAE":    float(mean_absolute_error(t, p)),
            "R2":     float(r2),
        })
    return pd.DataFrame(rows)


def assemble_full_prediction(
    pred_modeled: np.ndarray,
    X_ref: pd.DataFrame,
) -> pd.DataFrame:
    """
    Take a (n, 15) prediction array in MODEL_OUTPUT_COLS order and produce the
    full 19-column DataFrame by recovering T3, P3, P4 from X and
    MeOH_purity_stream4 from x_MeOH_4.
    """
    df = pd.DataFrame(pred_modeled, index=X_ref.index, columns=MODEL_OUTPUT_COLS)
    for out_col, in_col in DETERMINISTIC_FROM_INPUT.items():
        df[out_col] = X_ref[in_col].values
    for out_col, src_col in REDUNDANT_FROM_OUTPUT.items():
        df[out_col] = df[src_col].values
    return df[OUTPUT_COLS]


def regime_diagnostics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    no_liquid_mask: pd.Series | np.ndarray,
    columns: Iterable[str] = OUTPUT_COLS,
) -> pd.DataFrame:
    """All-rows / liquid-rows / no-liquid-rows aggregate metric breakdown."""
    if isinstance(no_liquid_mask, np.ndarray):
        no_liquid_mask = pd.Series(no_liquid_mask, index=y_true.index)
    masks = {
        "all_rows":       pd.Series(True, index=y_true.index),
        "liquid_rows":    ~no_liquid_mask,
        "no_liquid_rows": no_liquid_mask,
    }
    rows = []
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        m = regression_metrics_by_output(y_true.loc[mask], y_pred.loc[mask], columns)
        rows.append({
            "group":     name,
            "rows":      int(mask.sum()),
            "mean_RMSE": m["RMSE"].mean(),
            "mean_MAE":  m["MAE"].mean(),
            "mean_R2":   m["R2"].mean(skipna=True),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PINN constraint
# ---------------------------------------------------------------------------

def sum_to_one_constraint_loss(
    y_pred_scaled: torch.Tensor,
    Y_transformer: TargetTransformer,
    n4_total_true_raw: torch.Tensor,
    lambda_vapor: float = 1.0,
    lambda_liquid: float = 1.0,
) -> torch.Tensor:
    """
    Soft sum-to-one penalty in linear (un-scaled) space.

    Vapor (stream 3) is always present: penalize (sum of y_*_3 minus 1) squared.
    Liquid (stream 4) only when n4 > 0: same penalty on (sum of x_*_4 minus 1) squared.

    `y_pred_scaled` is the model output in standardized space.
    `n4_total_true_raw` is the raw n4_total ground truth for each row in the batch,
    used to gate the liquid penalty.
    """
    y_pred_lin = Y_transformer.inverse_transform_torch(y_pred_scaled)
    vapor_idx = [MODEL_OUTPUT_COLS.index(c) for c in VAPOR_FRACTION_COLS]
    liquid_idx = [MODEL_OUTPUT_COLS.index(c) for c in LIQUID_FRACTION_COLS]

    sum_v = y_pred_lin[:, vapor_idx].sum(dim=1)
    sum_l = y_pred_lin[:, liquid_idx].sum(dim=1)

    loss_v = ((sum_v - 1.0) ** 2).mean()
    gate = (n4_total_true_raw > 0).float()
    loss_l = (gate * (sum_l - 1.0) ** 2).sum() / gate.sum().clamp(min=1.0)

    return lambda_vapor * loss_v + lambda_liquid * loss_l


class VaporSumToOneProjection(nn.Module):
    """Hard linear-equality projection layer for the vapor mole fractions.

    Implements the KKT-hPINN idea (Chen et al., "Physics-Informed Neural
    Networks with Hard Linear Equality Constraints", arXiv:2402.07251): append a
    non-trainable layer that analytically projects the network output onto the
    constraint manifold, so the constraint holds *exactly* in both training and
    inference, not merely penalized.

    Constraint enforced:  y_CO_3 + y_H2_3 + y_MeOH_3 + y_H2O_3 = 1  (linear units).

    Because the vapor columns are NOT log-transformed (only ``x_H2_4`` is), the
    standardized -> linear map is affine, so the sum-to-one constraint is a
    single linear equality ``a^T z_v = c`` in standardized space with
        a_i = scale_i   (StandardScaler scale of each vapor column)
        c   = 1 - sum_i(mean_i)
    The L2-optimal projection (consistent with the scaled-space MSE loss) is the
    closed-form rank-1 correction
        z_v  <-  z_v - a * (a^T z_v - c) / (a^T a).
    No matrix inversion, no trainable parameters, no extra hyperparameters. Only
    the four vapor indices are modified; every other output passes through.
    """

    def __init__(self, vapor_idx: list[int], a: torch.Tensor, c: float):
        super().__init__()
        self.vapor_idx = list(vapor_idx)
        self.register_buffer("a", a.clone().detach().float())
        self.register_buffer("c", torch.tensor(float(c)))
        self.register_buffer("aa", (a * a).sum().float())

    @classmethod
    def from_transformer(cls, Y_transformer: "TargetTransformer") -> "VaporSumToOneProjection":
        """Build the projection from a *fitted* TargetTransformer's scaler params."""
        vapor_idx = [MODEL_OUTPUT_COLS.index(col) for col in VAPOR_FRACTION_COLS]
        assert not (set(vapor_idx) & set(Y_transformer.log_idx)), (
            "VaporSumToOneProjection assumes the vapor columns are affine-scaled "
            "(not log-transformed); the rank-1 projection is invalid otherwise."
        )
        mean = Y_transformer.scaler.mean_
        scale = Y_transformer.scaler.scale_
        a = torch.as_tensor(scale[vapor_idx], dtype=torch.float32)
        c = 1.0 - float(np.sum(mean[vapor_idx]))
        return cls(vapor_idx, a, c)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zv = z[:, self.vapor_idx]                        # (B, 4) standardized vapor
        correction = (zv @ self.a - self.c) / self.aa    # (B,)
        zv_proj = zv - correction.unsqueeze(1) * self.a  # (B, 4)
        z = z.clone()
        z[:, self.vapor_idx] = zv_proj
        return z


class LiquidClosure(nn.Module):
    """Hard, regime-gated sum-to-one closure for the liquid mole fractions.

    The liquid constraint x_CO_4 + x_H2_4 + x_MeOH_4 + x_H2O_4 = 1 holds *only*
    when a liquid phase exists (n4_total > 0); on no-liquid rows all four are 0.
    It is also *nonlinear* in the network's output space because ``x_H2_4`` is
    log-scaled, so the vapor rank-1 projection cannot be used (an orthogonal
    projection drives the tiny ``x_H2_4`` negative and destroys accuracy).
    Instead we apply the physically standard **closure (renormalisation)** in
    linear space, gated by the predicted regime:

        gate  = (n4_lin > n4_threshold)
        x     = clamp(liquid_lin, min=0)
        x_new = gate * x / sum(x)        # sum=1 on liquid rows, 0 on no-liquid rows

    Operates in standardized space (scaled in -> scaled out) so it slots straight
    after ``VaporSumToOneProjection`` in an ``nn.Sequential``. Non-trainable and
    differentiable (usable inside Task-4 optimisation). Self-contained via buffers
    so a checkpoint reloads without needing the ``TargetTransformer``.
    """

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor,
                 liquid_idx: list[int], log_idx: list[int], n4_idx: int,
                 n4_threshold: float = 0.05):
        super().__init__()
        self.register_buffer("mean", mean.clone().detach().float())
        self.register_buffer("scale", scale.clone().detach().float())
        self.liquid_idx = list(liquid_idx)
        self.log_idx = list(log_idx)
        self.n4_idx = int(n4_idx)
        self.n4_threshold = float(n4_threshold)

    @classmethod
    def from_transformer(cls, Y_transformer: "TargetTransformer",
                         n4_threshold: float = 0.05) -> "LiquidClosure":
        mean = torch.as_tensor(Y_transformer.scaler.mean_, dtype=torch.float32)
        scale = torch.as_tensor(Y_transformer.scaler.scale_, dtype=torch.float32)
        liquid_idx = [MODEL_OUTPUT_COLS.index(c) for c in LIQUID_FRACTION_COLS]
        n4_idx = MODEL_OUTPUT_COLS.index("n4_total")
        return cls(mean, scale, liquid_idx, list(Y_transformer.log_idx), n4_idx, n4_threshold)

    def _to_linear(self, z: torch.Tensor) -> torch.Tensor:
        lin = z * self.scale + self.mean
        if self.log_idx:
            lin = lin.clone()
            for j in self.log_idx:
                lin[:, j] = torch.exp(lin[:, j])
        return lin

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        lin = self._to_linear(z)
        n4 = lin[:, self.n4_idx]
        gate = (n4 > self.n4_threshold).float().unsqueeze(1)         # (B, 1)
        x = lin[:, self.liquid_idx].clamp(min=0.0)                   # (B, 4) linear
        s = x.sum(dim=1, keepdim=True).clamp(min=1e-12)
        x_new = gate * (x / s)                                       # sum=1 liquid, 0 no-liquid
        z = z.clone()
        for k, j in enumerate(self.liquid_idx):
            val = x_new[:, k]
            if j in self.log_idx:
                val = torch.log(val.clamp(min=LOG_FLOOR))
            z[:, j] = (val - self.mean[j]) / self.scale[j]
        return z


class EnsembleModel(nn.Module):
    """Average the (standardized-space) outputs of several member models.

    A deep ensemble: each member is an independently-trained network (e.g. the
    same architecture with different random seeds). Averaging their predictions
    reduces variance and almost always improves accuracy over any single member.

    Returns the mean of the members' outputs, so it is a drop-in ``nn.Module``
    that works with all the existing helpers (``predict_scaled``, ``train_model``)
    and chains straight into the hard constraint layers, e.g.
    ``nn.Sequential(EnsembleModel(members), VaporSumToOneProjection, LiquidClosure)``.
    """

    def __init__(self, members: Iterable[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(list(members))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members], dim=0).mean(dim=0)


def make_pinn_loss(
    Y_transformer: TargetTransformer,
    lambda_vapor: float = 0.0,
    lambda_liquid: float = 1.0,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a ``loss_fn(y_pred_scaled, y_true_scaled)`` for ``train_model``.

    Returns ``MSE(scaled) + sum_to_one_constraint_loss(...)``. The ``n4_total``
    gate for the regime-dependent liquid penalty is recovered from the *true*
    targets, so no DataLoader change is needed.

    - Hard-projection PINN: set ``lambda_vapor=0`` (the projection layer already
      enforces vapor sum-to-one exactly) and ``lambda_liquid>0``.
    - Soft-penalty ablation: set both ``lambda_vapor>0`` and ``lambda_liquid>0``.
    """
    n4_idx = MODEL_OUTPUT_COLS.index("n4_total")
    mse = nn.MSELoss()

    def loss_fn(y_pred_scaled: torch.Tensor, y_true_scaled: torch.Tensor) -> torch.Tensor:
        base = mse(y_pred_scaled, y_true_scaled)
        y_true_lin = Y_transformer.inverse_transform_torch(y_true_scaled)
        n4_true_raw = y_true_lin[:, n4_idx]
        penalty = sum_to_one_constraint_loss(
            y_pred_scaled, Y_transformer, n4_true_raw,
            lambda_vapor=lambda_vapor, lambda_liquid=lambda_liquid,
        )
        return base + penalty

    return loss_fn


# ---------------------------------------------------------------------------
# Parity plots
# ---------------------------------------------------------------------------

TRAIN_COLOR = "#4878d0"
TEST_COLOR = "#ee854a"


def parity_grid(
    y_train_true: pd.DataFrame,
    y_train_pred: pd.DataFrame,
    y_test_true: pd.DataFrame,
    y_test_pred: pd.DataFrame,
    no_liquid_train: pd.Series | np.ndarray,
    no_liquid_test: pd.Series | np.ndarray,
    columns: Iterable[str] = OUTPUT_COLS,
    title: str = "Parity plots",
    n_cols: int = 4,
) -> plt.Figure:
    """4-column parity-plot grid styled like XGBoost.ipynb's figure."""
    columns = list(columns)
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4))
    axes_flat = axes.flatten() if n_rows * n_cols > 1 else [axes]

    nl_tr = np.asarray(no_liquid_train).astype(bool)
    nl_te = np.asarray(no_liquid_test).astype(bool)
    liq_tr, liq_te = ~nl_tr, ~nl_te

    test_r2 = regression_metrics_by_output(y_test_true, y_test_pred, columns)
    r2_lookup = dict(zip(test_r2["output"], test_r2["R2"]))

    for j, col in enumerate(columns):
        ax = axes_flat[j]
        ytr_true = y_train_true[col].values
        ytr_pred = y_train_pred[col].values.astype(float)
        yte_true = y_test_true[col].values
        yte_pred = y_test_pred[col].values.astype(float)

        ax.scatter(ytr_true[liq_tr], ytr_pred[liq_tr], c=TRAIN_COLOR,
                   s=5, alpha=0.25, rasterized=True)
        if nl_tr.any():
            ax.scatter(ytr_true[nl_tr], ytr_pred[nl_tr], c=TRAIN_COLOR,
                       s=60, marker="x", linewidths=1.5, zorder=5)
        ax.scatter(yte_true[liq_te], yte_pred[liq_te], c=TEST_COLOR,
                   s=8, alpha=0.5, rasterized=True)
        if nl_te.any():
            ax.scatter(yte_true[nl_te], yte_pred[nl_te], c=TEST_COLOR,
                       s=60, marker="x", linewidths=1.5, zorder=5)

        all_vals = np.concatenate([ytr_true, ytr_pred, yte_true, yte_pred])
        lo, hi = all_vals.min(), all_vals.max()
        pad = max((hi - lo) * 0.05, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, zorder=3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{col}\ntest R²={r2_lookup.get(col, float('nan')):.4f}", fontsize=8)
        ax.set_xlabel("Actual", fontsize=7)
        ax.set_ylabel("Predicted", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(len(columns), len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TRAIN_COLOR,
               markersize=8, label="Train – liquid"),
        Line2D([0], [0], marker="x", color=TRAIN_COLOR, markersize=9,
               markeredgewidth=1.5, label="Train – no-liquid"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEST_COLOR,
               markersize=8, label="Test – liquid"),
        Line2D([0], [0], marker="x", color=TEST_COLOR, markersize=9,
               markeredgewidth=1.5, label="Test – no-liquid"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=9,
               bbox_to_anchor=(1.0, 0.0), framealpha=0.9)
    fig.suptitle(title, fontsize=11, y=1.005)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    scaled: ScaledData,
    history: TrainHistory,
    extra: dict | None = None,
) -> None:
    """Persist model weights + scalers + history so Task 4 can reload the surrogate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "X_scaler_mean": scaled.X_scaler.mean_,
        "X_scaler_scale": scaled.X_scaler.scale_,
        "Y_scaler_mean": scaled.Y_transformer.scaler.mean_,
        "Y_scaler_scale": scaled.Y_transformer.scaler.scale_,
        "log_idx": scaled.Y_transformer.log_idx,
        "input_cols": INPUT_COLS,
        "model_output_cols": MODEL_OUTPUT_COLS,
        "history": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "lr": history.lr,
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
        },
    }
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)
