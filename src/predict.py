"""Command-line entry point for training models and writing optional predictions.

Competition CSVs are expected to exist locally and are never copied into the
repository. By default this script only evaluates a 42-day historical holdout.
Use --predict-test to additionally write predictions for the competition test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_train_test_features
from model import BASELINE_CONFIG, SEASONAL_CONFIG, blend_predictions, predict, train_model
from validation import make_last_horizon_split, rmsle

SEED = 42
TARGET = "OrderVolume"


def _read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Celebal OrderVolume forecasting pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=42)
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--submission-path", type=Path, default=Path("submissions/predictions.csv"))
    parser.add_argument("--metrics-path", type=Path, default=Path("configs/metrics.json"))
    return parser.parse_args()


def _load_inputs(data_dir: Path, metadata_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "orders_train.csv"
    test_path = data_dir / "orders_test.csv"
    metadata_path = metadata_path or (data_dir / "hub_metadata.csv")
    for path in (train_path, test_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required local data file: {path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    metadata = pd.read_csv(metadata_path)
    return train, test, metadata


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {TARGET, "Date", "Id"}
    return [c for c in frame.columns if c not in excluded]


def evaluate_models(train: pd.DataFrame, metadata: pd.DataFrame, horizon: int) -> dict[str, float]:
    # Construct the validation boundary first. Features are then generated from
    # the chronological training observations; validation rows never contribute
    # target values to their own or later validation features.
    train = train.copy()
    train["Date"] = pd.to_datetime(train["Date"], errors="raise")
    split = make_last_horizon_split(train["Date"], horizon=horizon)
    cutoff = split.train_dates[-1]

    model_frame = train.loc[train["Date"] <= cutoff].copy()
    valid_frame = train.loc[train["Date"].isin(split.validation_dates)].copy()

    # Build features on the complete chronological frame but make the validation
    # simulation use only rows up through cutoff for target-derived features.
    # This preserves the causal information boundary.
    full = pd.concat([model_frame.assign(_part="train"), valid_frame.assign(_part="valid")], ignore_index=True)
    full = full.sort_values(["HubID", "Date"], kind="stable").reset_index(drop=True)
    train_feat, valid_feat = _causal_validation_features(full, model_frame, valid_frame, metadata)

    results: dict[str, float] = {}
    baseline_model, cols = train_model(train_feat, BASELINE_CONFIG, eval_frame=None)
    baseline_pred = predict(baseline_model, valid_feat, cols)
    results["baseline_rmsle"] = rmsle(valid_feat[TARGET], baseline_pred)

    seasonal_model, seasonal_cols = train_model(train_feat, SEASONAL_CONFIG, eval_frame=None)
    seasonal_pred = predict(seasonal_model, valid_feat, seasonal_cols)
    results["seasonal_rmsle"] = rmsle(valid_feat[TARGET], seasonal_pred)

    best_weight = 0.5
    best_score = float("inf")
    for weight in np.linspace(0.0, 1.0, 11):
        blended = blend_predictions(baseline_pred, seasonal_pred, weight_baseline=float(weight))
        score = rmsle(valid_feat[TARGET], blended)
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    results["blend_rmsle"] = best_score
    results["blend_baseline_weight"] = best_weight
    return results


def _causal_validation_features(
    full: pd.DataFrame,
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features for a holdout without exposing validation targets to training history."""
    train_dates = set(pd.to_datetime(train_rows["Date"]))
    valid_dates = set(pd.to_datetime(valid_rows["Date"]))

    # To avoid leakage, blank validation targets before feature construction. This
    # ensures lag/rolling calculations for validation only see observed train target
    # values before the cutoff. Calendar and static metadata remain available.
    causal = full.copy()
    causal.loc[causal["Date"].isin(valid_dates), TARGET] = np.nan
    train_part, valid_part = build_train_test_features(
        train_rows,
        causal.loc[causal["Date"].isin(valid_dates)].drop(columns=[TARGET], errors="ignore"),
        metadata,
    )
    # build_train_test_features expects test rows without a target and uses their
    # ordering. Restore validation targets strictly after features are computed.
    valid_part[TARGET] = valid_rows.sort_values(["HubID", "Date"], kind="stable")[TARGET].to_numpy()
    return train_part, valid_part


def run_prediction(train: pd.DataFrame, test: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train_feat, test_feat = build_train_test_features(train, test, metadata)

    baseline_model, baseline_cols = train_model(train_feat, BASELINE_CONFIG)
    baseline_pred = predict(baseline_model, test_feat, baseline_cols)

    seasonal_model, seasonal_cols = train_model(train_feat, SEASONAL_CONFIG)
    seasonal_pred = predict(seasonal_model, test_feat, seasonal_cols)

    # The final blend weight is selected on the historical holdout by evaluate_models.
    # A neutral fallback is used here; callers should overwrite from saved validation
    # metrics when running a final experiment.
    blend_pred = blend_predictions(baseline_pred, seasonal_pred, weight_baseline=0.5)
    out = test[[c for c in ("Id", "HubID", "Date") if c in test.columns]].copy()
    out[TARGET] = blend_pred
    return out, {
        "baseline_prediction_mean": float(np.mean(baseline_pred)),
        "seasonal_prediction_mean": float(np.mean(seasonal_pred)),
        "blend_prediction_mean": float(np.mean(blend_pred)),
    }


def main() -> None:
    args = _read_args()
    np.random.seed(SEED)
    train, test, metadata = _load_inputs(args.data_dir, args.metadata)

    metrics = evaluate_models(train, metadata, args.horizon)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if args.predict_test:
        predictions, summary = run_prediction(train, test, metadata)
        args.submission_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.submission_path, index=False)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
