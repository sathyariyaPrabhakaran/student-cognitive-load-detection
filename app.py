from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.feature_engineering import engineer_features, get_feature_columns

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
RESULTS_DIR = ROOT / "results"
FEATURE_PATH = ROOT / "models" / "feature_columns.joblib"
METADATA_PATH = RESULTS_DIR / "run_metadata.json"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"

app = Flask(__name__)

BASE_FEATURES = [
    "ear", "mar", "blink_count", "yawn_count", "head_movement",
    "typing_speed", "mouse_speed", "keyboard_idle", "mouse_idle",
    "mouse_clicks", "study_time"
]


def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def prediction_payload(frame, model):
    engineered = engineer_features(frame.copy())
    columns = get_feature_columns(include_derived=True)
    missing = [c for c in columns if c not in engineered.columns]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")

    X = engineered[columns].apply(pd.to_numeric, errors="coerce")
    if X.isna().all(axis=1).iloc[0]:
        raise ValueError("At least one valid numeric input is required.")

    prediction = str(model.predict(X)[0])
    response = {"cognitive_load": prediction}

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = [str(c) for c in model.classes_]
        response["probabilities"] = {
            label: round(float(prob), 4) for label, prob in zip(classes, probs)
        }
        response["confidence"] = round(float(max(probs)), 4)

    # Transparent signal interpretation. These are feature signals, not causal explanations.
    row = engineered.iloc[0]
    signals = []
    if float(row["idle_ratio"]) > 0.5:
        signals.append({"name": "Idle behaviour", "direction": "elevated", "detail": "Keyboard/mouse inactivity is relatively high."})
    if float(row["fatigue_proxy"]) > 1.0:
        signals.append({"name": "Fatigue proxy", "direction": "elevated", "detail": "Yawning, head movement and idle behaviour combine to a higher fatigue signal."})
    if float(row["activity_score"]) > 0:
        signals.append({"name": "Interaction activity", "direction": "present", "detail": "Typing, mouse movement and clicks contribute to the behavioural profile."})
    if float(row["engagement_proxy"]) > 0:
        signals.append({"name": "Engagement proxy", "direction": "present", "detail": "Interaction and blink activity are included in the engineered feature set."})
    response["signals"] = signals
    response["engineered_features"] = {
        c: round(float(row[c]), 4) if pd.notna(row[c]) else None
        for c in columns if c not in BASE_FEATURES
    }
    return response


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "dataset_available": (ROOT / "data" / "dataset.csv").exists(),
        "evaluation_available": COMPARISON_PATH.exists(),
    })


@app.get("/api/metadata")
def metadata():
    meta = read_json(METADATA_PATH)
    comparison = []
    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH).fillna(0).to_dict(orient="records")
    return jsonify({"metadata": meta, "comparison": comparison})


@app.post("/predict")
def predict():
    model = load_model()
    if model is None:
        return jsonify({"error": "No trained model is available. Add the dataset and run python src/train_models.py."}), 503

    payload = request.get_json(silent=True) or request.form.to_dict()
    frame = pd.DataFrame([payload])
    try:
        return jsonify(prediction_payload(frame, model))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
