from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_engineering import engineer_features, get_feature_columns
from temporal_features import add_session_dynamics, add_temporal_features

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
TARGET = "cognitive_load"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Add a real public/research dataset as data/dataset.csv before training.")
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    group_col = next((c for c in ["session_id", "participant_id", "subject_id", "user_id"] if c in df.columns), None)
    df = add_temporal_features(df, window=5, group_col=group_col)
    df = add_session_dynamics(df)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is required. Do not invent labels.")
    df = df.dropna(subset=[TARGET]).copy()
    if df[TARGET].nunique() < 2:
        raise ValueError("The target must contain at least two classes.")
    feature_candidates = [c for c in df.columns if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_candidates].replace([float("inf"), float("-inf")], pd.NA).apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(str)
    return X, y, feature_candidates, group_col


def build_models():
    return {
        "random_forest_baseline": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced"))]),
        "adaptive_gradient_boosting_proposed": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(learning_rate=0.06, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, random_state=42))]),
        "rbf_svm_comparator": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42))]),
    }


def evaluate():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    X, y, feature_columns, group_col = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, model in models.items():
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        rows.append({"model": name, "cv_macro_f1_mean": float(cv_f1.mean()), "cv_macro_f1_std": float(cv_f1.std()), "test_accuracy": float(accuracy_score(y_test, pred)), "test_macro_precision": float(precision), "test_macro_recall": float(recall), "test_macro_f1": float(f1)})
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(cm); ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(RESULTS_DIR / f"confusion_matrix_{name}.png", dpi=160); plt.close(fig)
        fitted[name] = model
    comparison = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    joblib.dump(fitted[best_name], MODELS_DIR / "best_model.joblib")
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.joblib")
    metadata = {"project_model": "Adaptive Temporal Behavioral Fusion", "best_model": best_name, "selection_metric": "5-fold cross-validation macro F1", "features": feature_columns, "target": TARGET, "n_samples": int(len(X)), "classes": sorted(y.unique().tolist()), "group_column": group_col, "data_source_required": "real research/public dataset; labels are never synthesized"}
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(comparison.to_string(index=False)); print(f"Selected model: {best_name}")


if __name__ == "__main__":
    evaluate()
