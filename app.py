from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.swell_adapter import prepare_swell
from src.temporal_features import add_session_dynamics, add_temporal_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
RESULTS_DIR = ROOT / "results"
FEATURE_PATH = ROOT / "models" / "feature_columns.joblib"
METADATA_PATH = RESULTS_DIR / "run_metadata.json"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
app = Flask(__name__)


def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def prepare_for_model(frame):
    model = load_model()
    if model is None:
        raise RuntimeError("No trained model. Run the SWELL setup and training pipeline first.")
    feature_columns = joblib.load(FEATURE_PATH)
    # Prediction input is expected to be already in the same raw feature schema as
    # the downloaded SWELL behavioural table. Recreate the temporal transformations.
    prepared, target, group_col = prepare_swell(frame.assign(condition="neutral"))
    prepared = add_temporal_features(prepared, window=5, group_col=group_col)
    prepared = add_session_dynamics(prepared)
    X = prepared[[c for c in feature_columns if c in prepared.columns]].apply(pd.to_numeric, errors="coerce")
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(f"Input session is missing {len(missing)} trained features. Upload the SWELL behavioural feature schema.")
    return model, X[feature_columns]


def predict_frame(frame):
    model, X = prepare_for_model(frame)
    predictions = model.predict(X)
    result = {"states": [str(v) for v in predictions]}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = [str(c) for c in model.classes_]
        result["classes"] = classes
        result["probabilities"] = [
            {c: round(float(p), 4) for c, p in zip(classes, row)} for row in probabilities
        ]
        result["mean_confidence"] = round(float(probabilities.max(axis=1).mean()), 4)
    return result


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "dataset_available": (ROOT / "data" / "swell_kw_behavioral.tab").exists(),
        "evaluation_available": COMPARISON_PATH.exists(),
    })


@app.get("/api/metadata")
def metadata():
    comparison = []
    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH).fillna(0).to_dict(orient="records")
    return jsonify({"metadata": read_json(METADATA_PATH), "comparison": comparison})


@app.post("/predict-session")
def predict_session():
    model = load_model()
    if model is None:
        return jsonify({"error": "No trained model is available. Run python scripts/setup_swell_kw.py and python src/train_models.py."}), 503
    upload = request.files.get("file")
    try:
        if upload:
            frame = pd.read_csv(upload, sep="\t")
        else:
            payload = request.get_json(silent=True) or {}
            rows = payload.get("rows", payload if isinstance(payload, list) else [])
            frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("Upload a non-empty SWELL-KW tab-delimited session file.")
        return jsonify(predict_frame(frame))
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
