"""Model training and prediction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("lightgbm is required. Install it with `pip install lightgbm`.") from exc


SEED = 42
TARGET = "OrderVolume"
DROP_COLUMNS = {TARGET, "Date", "Id"}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_estimators: int = 1400
    learning_rate: float = 0.035
    num_leaves: int = 63
    max_depth: int = -1
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.05
    reg_lambda: float = 0.5
    min_child_samples: int = 40
    seed: int = SEED


BASELINE_CONFIG = ModelConfig(name="baseline", n_estimators=1200, learning_rate=0.04, num_leaves=63)
SEASONAL_CONFIG = ModelConfig(name="seasonal", n_estimators=1600, learning_rate=0.03, num_leaves=95)


def _prepare_matrix(frame: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    X = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns], errors="ignore").copy()

    # LightGBM handles categorical columns well, but explicit pandas categorical
    # conversion makes the feature contract deterministic across train/test.
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")

    if feature_columns is None:
        feature_columns = list(X.columns)
    else:
        missing = [c for c in feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        X = X.reindex(columns=feature_columns)

    return X, list(feature_columns)


def _build_estimator(config: ModelConfig) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        min_child_samples=config.min_child_samples,
        random_state=config.seed,
        bagging_seed=config.seed,
        feature_fraction_seed=config.seed,
        data_random_seed=config.seed,
        n_jobs=-1,
        verbosity=-1,
    )


def train_model(
    train_frame: pd.DataFrame,
    config: ModelConfig,
    *,
    eval_frame: pd.DataFrame | None = None,
    early_stopping_rounds: int = 100,
) -> tuple[lgb.LGBMRegressor, list[str]]:
    """Train LightGBM on log1p(OrderVolume)."""
    if TARGET not in train_frame.columns:
        raise ValueError(f"Training frame must contain {TARGET!r}.")
    if (train_frame[TARGET] < 0).any():
        raise ValueError("OrderVolume must be non-negative.")

    X_train, feature_columns = _prepare_matrix(train_frame)
    y_train = np.log1p(train_frame[TARGET].astype(float).to_numpy())
    model = _build_estimator(config)

    fit_kwargs: dict = {}
    if eval_frame is not None:
        X_eval, _ = _prepare_matrix(eval_frame, feature_columns)
        y_eval = np.log1p(eval_frame[TARGET].astype(float).to_numpy())
        fit_kwargs = {
            "eval_set": [(X_eval, y_eval)],
            "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)],
        }

    model.fit(X_train, y_train, **fit_kwargs)
    return model, feature_columns


def predict(model: lgb.LGBMRegressor, frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    X, _ = _prepare_matrix(frame, feature_columns)
    pred = np.expm1(model.predict(X))
    pred = np.clip(pred, 0.0, None)
    if "IsOpen" in frame.columns:
        pred = np.where(frame["IsOpen"].to_numpy() == 0, 0.0, pred)
    return pred


def blend_predictions(
    baseline_pred: np.ndarray,
    seasonal_pred: np.ndarray,
    weight_baseline: float = 0.5,
) -> np.ndarray:
    if not 0.0 <= weight_baseline <= 1.0:
        raise ValueError("weight_baseline must be between 0 and 1.")
    baseline = np.asarray(baseline_pred, dtype=float)
    seasonal = np.asarray(seasonal_pred, dtype=float)
    if baseline.shape != seasonal.shape:
        raise ValueError("Prediction arrays must have the same shape.")
    return np.clip(weight_baseline * baseline + (1.0 - weight_baseline) * seasonal, 0.0, None)


__all__ = [
    "BASELINE_CONFIG",
    "SEASONAL_CONFIG",
    "ModelConfig",
    "blend_predictions",
    "predict",
    "train_model",
]
