"""Feature engineering for the Celebal OrderVolume forecasting task.

This module intentionally uses only columns present in the provided competition
files. Target-derived features are generated from observations available before
an observation date; no future target values are used.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TARGET = "OrderVolume"
DATE_COL = "Date"
HUB_COL = "HubID"

DEFAULT_TARGET_LAGS: tuple[int, ...] = (1, 7, 14, 28, 364, 371, 728, 735)
DEFAULT_SESSION_LAGS: tuple[int, ...] = (1, 7, 14, 28, 364, 371)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (7, 14, 28, 56)


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[DATE_COL], errors="raise")
    out["year"] = dt.dt.year.astype("int16")
    out["month"] = dt.dt.month.astype("int8")
    out["day"] = dt.dt.day.astype("int8")
    out["dayofweek"] = dt.dt.dayofweek.astype("int8")
    out["dayofyear"] = dt.dt.dayofyear.astype("int16")
    out["weekofyear"] = dt.dt.isocalendar().week.astype("int16")
    out["quarter"] = dt.dt.quarter.astype("int8")
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    out["time_idx"] = (dt - dt.min()).dt.days.astype("int32")
    return out


def _merge_metadata(
    orders: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    if metadata is None:
        return orders.copy()
    if HUB_COL not in metadata.columns:
        raise ValueError(f"Metadata must contain {HUB_COL!r}.")
    duplicate_ids = metadata[HUB_COL].duplicated().any()
    if duplicate_ids:
        raise ValueError("hub_metadata.csv must contain one row per HubID.")
    return orders.merge(metadata, on=HUB_COL, how="left", validate="many_to_one")


def _add_target_history(
    df: pd.DataFrame,
    lags: Sequence[int] = DEFAULT_TARGET_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    if TARGET not in df.columns:
        return df

    out = df.copy()
    grouped_target = out.groupby(HUB_COL, sort=False)[TARGET]
    for lag in lags:
        out[f"target_lag_{lag}"] = grouped_target.shift(lag)

    # Shift first, then roll, to ensure the current target and all future targets
    # are excluded from every rolling feature.
    shifted = grouped_target.shift(1)
    for window in rolling_windows:
        rolled = shifted.groupby(out[HUB_COL], sort=False).rolling(window, min_periods=1)
        stats = rolled.agg(["mean", "std", "min", "max"]).reset_index(level=0, drop=True)
        out[f"target_roll_mean_{window}"] = stats["mean"].to_numpy()
        out[f"target_roll_std_{window}"] = stats["std"].to_numpy()
        out[f"target_roll_min_{window}"] = stats["min"].to_numpy()
        out[f"target_roll_max_{window}"] = stats["max"].to_numpy()

    # Same-weekday historical signal. Only previous observations are eligible.
    day_name = out["dayofweek"]
    temp = pd.DataFrame({HUB_COL: out[HUB_COL], "dow": day_name, TARGET: out[TARGET]})
    temp["shifted_target"] = temp.groupby([HUB_COL, "dow"], sort=False)[TARGET].shift(1)
    same_dow = temp.groupby([HUB_COL, "dow"], sort=False)["shifted_target"].transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    out["target_same_dow_expanding_mean"] = same_dow.to_numpy()
    return out


def _add_session_history(
    df: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_SESSION_LAGS,
) -> pd.DataFrame:
    """Add historical AppSessions features.

    AppSessions is deliberately never referenced contemporaneously for test rows,
    because the competition test set does not provide that field.
    """
    if "AppSessions" not in df.columns:
        return df
    out = df.copy()
    grouped_sessions = out.groupby(HUB_COL, sort=False)["AppSessions"]
    for lag in lags:
        out[f"sessions_lag_{lag}"] = grouped_sessions.shift(lag)
    shifted = grouped_sessions.shift(1)
    for window in (7, 28):
        roll = shifted.groupby(out[HUB_COL], sort=False).rolling(window, min_periods=1).mean()
        out[f"sessions_roll_mean_{window}"] = roll.reset_index(level=0, drop=True).to_numpy()
    return out


def _add_metadata_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[DATE_COL], errors="raise")

    if {"CompetitorOpenSinceMonth", "CompetitorOpenSinceYear"}.issubset(out.columns):
        comp_date = pd.to_datetime(
            {
                "year": pd.to_numeric(out["CompetitorOpenSinceYear"], errors="coerce"),
                "month": pd.to_numeric(out["CompetitorOpenSinceMonth"], errors="coerce"),
                "day": 1,
            },
            errors="coerce",
        )
        out["competitor_age_days"] = (dt - comp_date).dt.days

    if {"LoyaltyProgramSinceYear", "LoyaltyProgramSinceWeek"}.issubset(out.columns):
        year = pd.to_numeric(out["LoyaltyProgramSinceYear"], errors="coerce")
        week = pd.to_numeric(out["LoyaltyProgramSinceWeek"], errors="coerce")
        iso = pd.DataFrame({"year": year, "week": week, "day": 1})
        loyalty_date = pd.to_datetime(iso, errors="coerce")
        out["loyalty_age_days"] = (dt - loyalty_date).dt.days

    if "LoyaltyProgramInterval" in out.columns:
        month_num = out["month"]
        interval = out["LoyaltyProgramInterval"].fillna("").astype(str)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for idx, name in enumerate(month_names, start=1):
            out[f"loyalty_round_month_{idx}"] = (
                interval.str.contains(name, regex=False) & (month_num == idx)
            ).astype("int8")

    return out


def make_features(
    orders: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    *,
    include_target_history: bool = True,
) -> pd.DataFrame:
    """Create deterministic forecasting features.

    The input must contain HubID and Date. For training data, target history and
    historical AppSessions features may be generated. For test data, those
    features must be supplied through a combined train+test frame when needed.
    """
    required = {HUB_COL, DATE_COL}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = _merge_metadata(orders.copy(), metadata)
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="raise")
    out = out.sort_values([HUB_COL, DATE_COL], kind="stable").reset_index(drop=True)
    out = _add_calendar_features(out)
    out = _add_metadata_time_features(out)
    if include_target_history:
        out = _add_target_history(out)
        out = _add_session_history(out)

    # Training and test need identical feature columns when they are passed to a
    # model. ID-like fields are retained in the frame for downstream bookkeeping.
    return out


def build_train_test_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leakage-safe train/test features from a single chronological frame.

    Target history is computed on train only. For test rows, target lags therefore
    reference only historical train observations. AppSessions history is computed
    from train plus test features only where historical values exist; no current or
    future test AppSessions is consumed because that column is absent from test.
    """
    train_local = train.copy()
    test_local = test.copy()
    train_local["_is_train"] = 1
    test_local["_is_train"] = 0
    test_local[TARGET] = np.nan

    combined = pd.concat([train_local, test_local], ignore_index=True, sort=False)
    combined = _merge_metadata(combined, metadata)
    combined[DATE_COL] = pd.to_datetime(combined[DATE_COL], errors="raise")
    combined = combined.sort_values([HUB_COL, DATE_COL], kind="stable").reset_index(drop=True)

    combined = _add_calendar_features(combined)
    combined = _add_metadata_time_features(combined)

    # Compute lag history using the known training target only. NaN test targets
    # naturally propagate into test lags unless the lag points back into train.
    combined = _add_target_history(combined)

    # For sessions, first preserve train values and blank all test values so no
    # contemporaneous/future test AppSessions can leak into features.
    if "AppSessions" in combined.columns:
        combined.loc[combined["_is_train"] == 0, "AppSessions"] = np.nan
    combined = _add_session_history(combined)

    train_feat = combined.loc[combined["_is_train"] == 1].drop(columns=["_is_train"]).copy()
    test_feat = combined.loc[combined["_is_train"] == 0].drop(columns=["_is_train", TARGET]).copy()
    return train_feat, test_feat


__all__ = [
    "DEFAULT_TARGET_LAGS",
    "DEFAULT_SESSION_LAGS",
    "DEFAULT_ROLLING_WINDOWS",
    "build_train_test_features",
    "make_features",
]
