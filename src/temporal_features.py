"""Temporal feature extraction for session-aware cognitive-load modelling."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame, window: int = 5, group_col: str | None = None) -> pd.DataFrame:
    """Add rolling level, variability and trend features without using future rows.

    If a participant/session column exists, windows are reset per group. Otherwise
    the input order is treated as the session order.
    """
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"cognitive_load", "target", "label"}
    numeric = [c for c in numeric if c not in excluded]
    groups = out.groupby(group_col, sort=False) if group_col and group_col in out.columns else [(None, out)]

    for col in numeric:
        pieces = []
        for _, g in groups:
            s = pd.to_numeric(g[col], errors="coerce")
            roll = s.rolling(window=window, min_periods=1)
            tmp = pd.DataFrame(index=g.index)
            tmp[f"{col}_roll_mean"] = roll.mean()
            tmp[f"{col}_roll_std"] = roll.std().fillna(0)
            tmp[f"{col}_delta"] = s.diff().fillna(0)
            tmp[f"{col}_trend"] = s.diff(periods=min(window - 1, max(1, len(g) - 1))).fillna(0)
            pieces.append(tmp)
        derived = pd.concat(pieces).sort_index()
        out = pd.concat([out, derived], axis=1)
    return out


def add_session_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Create robust aggregate dynamics used by the adaptive fusion layer."""
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    interaction = [c for c in numeric if any(k in c.lower() for k in ("typing", "mouse", "click", "key"))]
    fatigue = [c for c in numeric if any(k in c.lower() for k in ("idle", "yawn", "fatigue"))]
    if interaction:
        out["interaction_intensity"] = out[interaction].fillna(0).mean(axis=1)
    if fatigue:
        out["fatigue_pressure"] = out[fatigue].fillna(0).mean(axis=1)
    if "interaction_intensity" in out and "fatigue_pressure" in out:
        out["adaptive_balance"] = out["interaction_intensity"] / (1.0 + out["fatigue_pressure"].abs())
    return out
