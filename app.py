from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.feature_engineering import engineer_features, get_feature_columns

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return render_template("index.html") if (ROOT / "templates" / "index.html").exists() else jsonify({"service": "Student Cognitive Load Detection", "status": "ready"})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_available": MODEL_PATH.exists()})


@app.post("/predict")
def predict():
    model = load_model()
    if model is None:
        return jsonify({"error": "Trained model not found. Run src/train_models.py after adding data/dataset.csv."}), 503
    payload = request.get_json(silent=True) or request.form.to_dict()
    frame = pd.DataFrame([payload])
    frame = engineer_features(frame)
    columns = get_feature_columns(include_derived=True)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        return jsonify({"error": "Missing features", "missing": missing}), 400
    X = frame[columns].apply(pd.to_numeric, errors="coerce")
    prediction = model.predict(X)[0]
    response = {"cognitive_load": str(prediction)}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        response["confidence"] = float(max(probs))
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
