"""SWELL-KW adapter and leakage-safe workload-state construction."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_swell(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Normalize metadata headers for robust matching across DANS exports.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lowered = {c.lower().replace(" ", ""): c for c in df.columns}
    for n in names:
        key = n.lower().replace(" ", "")
        if key in lowered:
            return lowered[key]
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if any(n.lower().replace(" ", "") in cl for n in names):
            return c
    return None


def prepare_swell(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str | None]:
    """Return numeric features and a transparent workload-state target.

    SWELL-KW records experimental working conditions (neutral, interruption,
    time-pressure and relaxation). These are workload-condition labels, not
    clinical diagnoses. The model therefore reports *workload state* and the
    dashboard can describe it as an inferred cognitive-load proxy.
    """
    condition = find_column(df, ["condition", "block", "state", "workingcondition"])
    participant = find_column(df, ["participant", "subject", "user", "person"])
    if condition is None:
        raise ValueError("SWELL file has no recognizable condition column.")

    out = df.copy()
    raw = out[condition].astype(str).str.strip().str.lower()
    mapping = {}
    for value in sorted(raw.dropna().unique()):
        if value in {"r", "relax", "relaxation"}:
            mapping[value] = "relaxed"
        elif value in {"n", "neutral", "normal"}:
            mapping[value] = "baseline"
        elif value in {"i", "interruptions", "interruption", "email interruption"}:
            mapping[value] = "elevated"
        elif value in {"t", "time pressure", "timepressure"}:
            mapping[value] = "high"
    out["workload_state"] = raw.map(mapping)
    out = out.dropna(subset=["workload_state"]).copy()

    # Preserve participant/session ordering for temporal windows while removing
    # labels and obvious identifiers from the feature matrix.
    exclude = {condition, "workload_state"}
    if participant:
        exclude.add(participant)
    numeric = [c for c in out.columns if c not in exclude and pd.api.types.is_numeric_dtype(out[c])]
    if not numeric:
        raise ValueError("No numeric SWELL behavioural features were detected.")
    result = out[numeric + ["workload_state"]].copy()
    if participant:
        result["participant_id"] = out[participant].astype(str).values
    return result, "workload_state", "participant_id" if participant else None
