"""Leakage-safe time-based validation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex


def make_last_horizon_split(
    dates: pd.Series | pd.DatetimeIndex,
    horizon: int = 42,
) -> TimeSplit:
    """Return the final `horizon` unique dates as validation dates."""
    unique_dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates)).dropna().unique()).sort_values()
    if len(unique_dates) <= horizon:
        raise ValueError("Not enough unique dates for the requested validation horizon.")
    return TimeSplit(
        train_dates=unique_dates[:-horizon],
        validation_dates=unique_dates[-horizon:],
    )


def rmsle(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Compute RMSLE safely for non-negative regression targets."""
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if actual.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(actual < 0):
        raise ValueError("RMSLE requires non-negative targets.")
    pred = np.clip(pred, 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(actual)) ** 2)))


def split_by_dates(
    frame: pd.DataFrame,
    split: TimeSplit,
    date_col: str = "Date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into training and validation rows by calendar date."""
    dates = pd.to_datetime(frame[date_col], errors="raise")
    train = frame.loc[dates.isin(split.train_dates)].copy()
    valid = frame.loc[dates.isin(split.validation_dates)].copy()
    if train.empty or valid.empty:
        raise ValueError("Date split produced an empty partition.")
    return train, valid


def assert_no_future_target_features(
    feature_frame: pd.DataFrame,
    date_col: str = "Date",
    target: str = "OrderVolume",
) -> None:
    """Basic guardrail: target-derived feature names must be lag/rolling only.

    This is not a substitute for a careful feature audit, but it catches accidental
    inclusion of the raw target under an unexpected feature name.
    """
    forbidden = {target.lower()}
    suspicious = []
    for col in feature_frame.columns:
        lower = str(col).lower()
        if lower in forbidden and col != target:
            suspicious.append(col)
    if suspicious:
        raise AssertionError(f"Suspicious target feature columns: {suspicious}")


__all__ = ["TimeSplit", "assert_no_future_target_features", "make_last_horizon_split", "rmsle", "split_by_dates"]
