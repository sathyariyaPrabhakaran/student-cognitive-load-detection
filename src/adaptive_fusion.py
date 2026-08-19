"""Adaptive fusion of behavioural, fatigue and interaction evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd


def feature_group_weights(df: pd.DataFrame) -> dict[str, float]:
    """Estimate normalized evidence weights from available feature groups.

    This is an interpretable feature-fusion layer, not a claim of causal importance.
    """
    cols = {c.lower(): c for c in df.columns}
    groups = {
        "interaction": [c for c in df.columns if any(k in c.lower() for k in ("typing", "mouse", "click", "interaction"))],
        "fatigue": [c for c in df.columns if any(k in c.lower() for k in ("yawn", "idle", "fatigue", "head"))],
        "facial": [c for c in df.columns if any(k in c.lower() for k in ("ear", "mar", "blink"))],
        "temporal": [c for c in df.columns if any(k in c.lower() for k in ("roll_", "_delta", "_trend", "balance"))],
    }
    raw = {}
    for name, members in groups.items():
        if not members:
            raw[name] = 0.0
            continue
        values = df[members].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        raw[name] = float(np.mean(np.abs(values.values)))
    total = sum(raw.values()) or 1.0
    return {k: round(v / total, 4) for k, v in raw.items()}


def fused_evidence(df: pd.DataFrame) -> pd.Series:
    """Return an interpretable normalized evidence score for dashboard use."""
    weights = feature_group_weights(df)
    score = pd.Series(0.0, index=df.index)
    for group, weight in weights.items():
        members = [c for c in df.columns if any(k in c.lower() for k in {
            "interaction": ["typing", "mouse", "click", "interaction"],
            "fatigue": ["yawn", "idle", "fatigue", "head"],
            "facial": ["ear", "mar", "blink"],
            "temporal": ["roll_", "_delta", "_trend", "balance"],
        }[group])]
        if members:
            score += weight * df[members].apply(pd.to_numeric, errors="coerce").fillna(0).abs().mean(axis=1)
    return score
