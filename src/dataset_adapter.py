"""Adapters for converting public cognitive-workload data into project features.

The project accepts a canonical dataset.csv schema. This module provides a
transparent adapter for eye-tracking datasets and documents which features are
observed versus derived. It never fabricates measurements.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dataset.csv"


def adapt_eye_tracking(input_csv: str, output_csv: str | None = None) -> pd.DataFrame:
    """Adapt a prepared eye-tracking CSV to the canonical schema.

    Expected input columns are the project-compatible measurements listed below.
    Dataset-specific extraction should happen before this adapter; absent
    measurements remain NaN and are handled by the training pipeline.
    """
    df = pd.read_csv(input_csv)
    mapping = {
        "ear": ["ear", "eye_aspect_ratio"],
        "mar": ["mar", "mouth_aspect_ratio"],
        "blink_count": ["blink_count", "blinks"],
        "yawn_count": ["yawn_count", "yawns"],
        "head_movement": ["head_movement", "head_motion"],
        "typing_speed": ["typing_speed", "keystrokes_per_minute"],
        "mouse_speed": ["mouse_speed"],
        "keyboard_idle": ["keyboard_idle"],
        "mouse_idle": ["mouse_idle"],
        "mouse_clicks": ["mouse_clicks", "click_count"],
        "study_time": ["study_time", "duration", "task_duration"],
        "cognitive_load": ["cognitive_load", "load", "workload", "label"],
    }
    out = pd.DataFrame(index=df.index)
    for target, candidates in mapping.items():
        source = next((c for c in candidates if c in df.columns), None)
        out[target] = df[source] if source else pd.NA
    if out["cognitive_load"].isna().all():
        raise ValueError("No cognitive-load label found. Map the dataset's official workload label to cognitive_load.")
    destination = Path(output_csv) if output_csv else OUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)
    return out
