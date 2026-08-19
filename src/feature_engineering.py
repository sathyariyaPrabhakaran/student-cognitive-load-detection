import pandas as pd

BASE_FEATURES = [
    "ear", "mar", "blink_count", "yawn_count", "head_movement",
    "typing_speed", "mouse_speed", "keyboard_idle", "mouse_idle",
    "mouse_clicks", "study_time"
]

DERIVED_FEATURES = [
    "blink_yawn_ratio", "activity_score", "idle_ratio",
    "interaction_rate", "fatigue_proxy", "engagement_proxy"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stable behavioral features from the project's 11 base features."""
    out = df.copy()
    eps = 1e-6
    numeric = [c for c in BASE_FEATURES if c in out.columns]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["blink_yawn_ratio"] = out["blink_count"] / (out["yawn_count"] + eps)
    out["activity_score"] = out["typing_speed"] + out["mouse_speed"] + out["mouse_clicks"]
    out["idle_ratio"] = (out["keyboard_idle"] + out["mouse_idle"]) / 2.0
    out["interaction_rate"] = (out["typing_speed"] + out["mouse_speed"]) / (out["study_time"] + eps)
    out["fatigue_proxy"] = out["yawn_count"] + out["head_movement"] + out["idle_ratio"]
    out["engagement_proxy"] = out["typing_speed"] + out["mouse_speed"] + out["blink_count"]
    return out


def get_feature_columns(include_derived: bool = True):
    return BASE_FEATURES + (DERIVED_FEATURES if include_derived else [])
