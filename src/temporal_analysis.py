from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset.csv"
RESULTS_DIR = ROOT / "results"


def run_temporal_analysis():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    time_col = next((c for c in ["timestamp", "datetime", "time", "created_at"] if c in df.columns), None)
    if time_col is None:
        return {"status": "skipped", "reason": "No timestamp/datetime column in dataset."}

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).copy()
    if df.empty:
        return {"status": "skipped", "reason": "Timestamp column contains no valid values."}

    if "cognitive_load" in df.columns:
        summary = df.groupby(df[time_col].dt.hour)["cognitive_load"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown")
        RESULTS_DIR.mkdir(exist_ok=True)
        summary.rename("dominant_cognitive_load").to_csv(RESULTS_DIR / "temporal_load_summary.csv")

    return {"status": "completed", "rows_with_valid_time": int(len(df)), "time_column": time_col}


if __name__ == "__main__":
    print(run_temporal_analysis())
