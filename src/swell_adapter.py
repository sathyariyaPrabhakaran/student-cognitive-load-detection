"""Robust SWELL-KW reader and workload-state adapter."""
from __future__ import annotations
from pathlib import Path
import csv
import pandas as pd


def _detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:262144]
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            pass
    return "latin1"


def _detect_separator(path: Path, encoding: str) -> str:
    sample = path.read_text(encoding=encoding, errors="replace")[:10000]
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;|").delimiter
    except csv.Error:
        counts = {s: sample.count(s) for s in ("\t", ",", ";", "|")}
        return max(counts, key=counts.get)


def read_swell(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SWELL dataset not found: {path}")
    encoding = _detect_encoding(path)
    separator = _detect_separator(path, encoding)
    print(f"[SWELL] encoding={encoding}, separator={separator!r}")
    df = pd.read_csv(path, sep=separator, encoding=encoding, low_memory=False, on_bad_lines="warn")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if df.empty:
        raise ValueError("The SWELL file was read successfully but contains no rows.")
    print(f"[SWELL] shape={df.shape}")
    print(f"[SWELL] first columns={list(df.columns[:12])}")
    return df


def find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for n in names:
        key = n.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    for c in df.columns:
        cl = c.lower().replace(" ", "").replace("_", "")
        if any(n.lower().replace(" ", "").replace("_", "") in cl for n in names):
            return c
    return None


def prepare_swell(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str | None]:
    condition = find_column(df, ["condition", "block", "state", "workingcondition"])
    participant = find_column(df, ["participant", "subject", "user", "person"])
    if condition is None:
        raise ValueError("SWELL file has no recognizable condition column. Inspect the printed columns above before training.")
    out = df.copy()
    raw = out[condition].astype(str).str.strip().str.lower()
    mapping = {}
    for value in sorted(raw.dropna().unique()):
        if value in {"r", "relax", "relaxation"}: mapping[value] = "relaxed"
        elif value in {"n", "neutral", "normal"}: mapping[value] = "baseline"
        elif value in {"i", "interruptions", "interruption", "email interruption"}: mapping[value] = "elevated"
        elif value in {"t", "time pressure", "timepressure"}: mapping[value] = "high"
    out["workload_state"] = raw.map(mapping)
    out = out.dropna(subset=["workload_state"]).copy()
    exclude = {condition, "workload_state"}
    if participant: exclude.add(participant)
    numeric = [c for c in out.columns if c not in exclude and pd.api.types.is_numeric_dtype(out[c])]
    if not numeric: raise ValueError("No numeric SWELL behavioural features were detected.")
    result = out[numeric + ["workload_state"]].copy()
    if participant: result["participant_id"] = out[participant].astype(str).values
    return result, "workload_state", "participant_id" if participant else None
