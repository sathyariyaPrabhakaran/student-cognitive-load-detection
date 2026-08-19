"""SWELL-KW loader, label adapter and leakage-aware feature selection."""
from __future__ import annotations
from pathlib import Path
import csv, gzip, tarfile, tempfile, zipfile
import pandas as pd


def _detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:262144]
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try: raw.decode(enc); return enc
        except UnicodeDecodeError: pass
    return "latin1"


def _detect_separator(path: Path, encoding: str) -> str:
    sample = path.read_text(encoding=encoding, errors="replace")[:10000]
    try: return csv.Sniffer().sniff(sample, delimiters="\t,;|").delimiter
    except csv.Error:
        counts = {s: sample.count(s) for s in ("\t", ",", ";", "|")}
        return max(counts, key=counts.get)


def _archive_kind(path: Path) -> str | None:
    with path.open("rb") as f: head = f.read(512)
    if head.startswith(b"PK\x03\x04"): return "zip"
    if head.startswith(b"\x1f\x8b"): return "gzip"
    if len(head) >= 262 and head[257:262] == b"ustar": return "tar"
    return None


def _is_excel(path: Path) -> bool:
    if _archive_kind(path) != "zip": return False
    try:
        with zipfile.ZipFile(path) as z: names = set(z.namelist())
        return "[Content_Types].xml" in names and "xl/workbook.xml" in names
    except zipfile.BadZipFile: return False


def _extract(path: Path) -> Path:
    if _is_excel(path):
        print("[SWELL] detected OOXML Excel workbook container"); return path
    kind = _archive_kind(path)
    if not kind: return path
    out = Path(tempfile.mkdtemp(prefix="swell_"))
    if kind == "zip":
        with zipfile.ZipFile(path) as z: z.extractall(out)
    elif kind == "gzip":
        target = out / path.stem
        with gzip.open(path, "rb") as src, target.open("wb") as dst: dst.write(src.read())
    else:
        with tarfile.open(path) as t: t.extractall(out, filter="data")
    files = [p for p in out.rglob("*") if p.is_file()]
    preferred = [p for p in files if p.suffix.lower() in {".tab", ".txt", ".csv", ".tsv", ".xlsx", ".xls"}]
    files = preferred or files
    files.sort(key=lambda p: (not any(k in p.name.lower() for k in ("behavior", "behaviour", "swell", "workload", "feature")), -p.stat().st_size))
    chosen = files[0]
    print(f"[SWELL] extracted archive -> {chosen.name}")
    return chosen


def read_swell(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"SWELL dataset not found: {path}")
    actual = _extract(path)
    if _is_excel(actual):
        try: df = pd.read_excel(actual, engine="openpyxl")
        except ImportError as e: raise ImportError("Run: python -m pip install openpyxl") from e
        print(f"[SWELL] Excel sheet shape={df.shape}")
    else:
        enc = _detect_encoding(actual); sep = _detect_separator(actual, enc)
        print(f"[SWELL] encoding={enc}, separator={sep!r}")
        df = pd.read_csv(actual, sep=sep, encoding=enc, engine="python", on_bad_lines="warn")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if df.empty: raise ValueError("The SWELL dataset contains no data rows.")
    print(f"[SWELL] shape={df.shape}")
    print(f"[SWELL] first columns={list(df.columns[:12])}")
    return df


def find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    norm = {str(c).lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for n in names:
        k = n.lower().replace(" ", "").replace("_", "")
        if k in norm: return norm[k]
    return None


def prepare_swell(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    condition = find_column(df, ["condition", "workingcondition", "workloadcondition"])
    participant = find_column(df, ["pp", "participant", "subject", "user", "person"])
    if condition is None: raise ValueError("No Condition column found in SWELL data.")
    if participant is None: raise ValueError("No participant column found; expected PP/participant/subject.")
    out = df.copy()
    raw = out[condition].astype(str).str.strip().str.lower()
    print(f"[SWELL] condition classes={sorted(raw.unique().tolist())}")
    labels = {
        "r":"relaxed", "relax":"relaxed", "relaxation":"relaxed",
        "b":"baseline", "n":"baseline", "neutral":"baseline", "normal":"baseline",
        "i":"elevated", "interruptions":"elevated", "interruption":"elevated", "email interruption":"elevated",
        "t":"high", "time pressure":"high", "timepressure":"high"
    }
    out["workload_state"] = raw.map(labels).fillna(raw)
    out["participant_id"] = out[participant].astype(str)

    # Exclude subjective/self-report outcomes so the model must infer workload from signals.
    blocked = ("mentaleffort", "mentaldemand", "physicaldemand", "temporaldemand", "performance", "frustration", "stress_rc", "valence_rc", "arousal_rc", "dominance")
    numeric = []
    for c in out.columns:
        key = str(c).lower().replace(" ", "").replace("_", "")
        if c in {condition, participant, "workload_state", "participant_id"}: continue
        if any(token.replace("_", "") in key for token in blocked): continue
        if pd.api.types.is_numeric_dtype(out[c]): numeric.append(c)
    if not numeric: raise ValueError("No leakage-safe numeric features were found.")
    result = out[numeric + ["workload_state", "participant_id"]].copy()
    print(f"[SWELL] leakage-safe numeric features={len(numeric)}")
    return result, "workload_state", "participant_id"
