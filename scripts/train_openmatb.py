"""Train the project models on a reproducible public workload benchmark.

Source: MInD-Laboratory/Measuring_Workload_Dynamics_in_OpenMATB.
The source contains performance-derived features and H/M/L workload conditions.
This benchmark is intentionally separate from the webcam deployment model because
performance features cannot be truthfully reconstructed from a webcam alone.
"""
from pathlib import Path
import json
import urllib.request

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "openmatb_performance.csv"
RESULTS = ROOT / "results" / "openmatb"
MODELS = ROOT / "models" / "openmatb"
URL = "https://raw.githubusercontent.com/MInD-Laboratory/Measuring_Workload_Dynamics_in_OpenMATB/main/Modeling/baseline_features/performance_baseline.csv"


def download():
    DATA.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, DATA)


def main():
    download()
    df = pd.read_csv(DATA)
    df = df[df["condition"].isin(["L", "M", "H"])].copy()
    y = df.pop("condition").map({"L": "low", "M": "medium", "H": "high"})
    df = df.drop(columns=["participant", "window_index"], errors="ignore")
    X = df.select_dtypes(include="number")
    if X.empty or y.nunique() < 3:
        raise RuntimeError("Public OpenMATB benchmark did not provide the expected numeric features/classes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
    models = {
        "random_forest_baseline": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1, class_weight="balanced"))]),
        "hist_gradient_boosting_proposed": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(learning_rate=.08, max_iter=250, max_leaf_nodes=15, l2_regularization=.5, random_state=42))]),
        "rbf_svm": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVC(C=2, kernel="rbf", probability=True, class_weight="balanced", random_state=42))]),
    }
    RESULTS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []; fitted = {}
    for name, model in models.items():
        cvf1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        model.fit(X_train, y_train); pred = model.predict(X_test)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        rows.append({"model": name, "cv_macro_f1_mean": cvf1.mean(), "cv_macro_f1_std": cvf1.std(), "test_accuracy": accuracy_score(y_test, pred), "test_macro_precision": p, "test_macro_recall": r, "test_macro_f1": f1})
        labels = ["low", "medium", "high"]
        cm = confusion_matrix(y_test, pred, labels=labels)
        plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(name); plt.tight_layout(); plt.savefig(RESULTS / f"confusion_matrix_{name}.png", dpi=160); plt.close()
        fitted[name] = model
    comparison = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
    best = comparison.iloc[0]["model"]
    joblib.dump(fitted[best], MODELS / "best_model.joblib")
    joblib.dump(X.columns.tolist(), MODELS / "feature_columns.joblib")
    (RESULTS / "run_metadata.json").write_text(json.dumps({"source": URL, "best_model": best, "selection_metric": "5-fold cross-validation macro F1", "n_samples": len(X), "features": X.columns.tolist(), "classes": sorted(y.unique().tolist())}, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
