from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

FEATURE_COLUMNS = [
    "ear", "mar", "blink_count", "yawn_count", "head_movement",
    "typing_speed", "mouse_speed", "keyboard_idle", "mouse_idle",
    "mouse_clicks", "study_time"
]
TARGET = "cognitive_load"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place the real dataset.csv in data/."
        )
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=[TARGET]).copy()
    if df[TARGET].nunique() < 2:
        raise ValueError("The target must contain at least two classes.")
    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(str)
    return X, y


def build_models():
    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42,
            n_jobs=-1, class_weight="balanced"
        )),
    ])
    proposed = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=250, max_leaf_nodes=15,
            l2_regularization=0.5, random_state=42
        )),
    ])
    svm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(C=2.0, kernel="rbf", probability=True,
                      class_weight="balanced", random_state=42)),
    ])
    return {
        "random_forest_baseline": baseline,
        "hist_gradient_boosting_proposed": proposed,
        "rbf_svm": svm,
    }


def evaluate():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}

    for name, model in models.items():
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="f1_macro")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="macro", zero_division=0
        )
        rows.append({
            "model": name,
            "cv_macro_f1_mean": cv_f1.mean(),
            "cv_macro_f1_std": cv_f1.std(),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_macro_precision": precision,
            "test_macro_recall": recall,
            "test_macro_f1": f1,
        })
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, pred, labels=labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"confusion_matrix_{name}.png", dpi=160)
        plt.close()
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    joblib.dump(fitted[best_name], MODELS_DIR / "best_model.joblib")
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "feature_columns.joblib")
    metadata = {
        "best_model": best_name,
        "selection_metric": "5-fold cross-validation macro F1",
        "features": FEATURE_COLUMNS,
        "target": TARGET,
        "n_samples": int(len(X)),
        "classes": sorted(y.unique().tolist()),
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(comparison.to_string(index=False))
    print(f"Selected model: {best_name}")


if __name__ == "__main__":
    evaluate()
