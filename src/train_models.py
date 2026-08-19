from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from swell_adapter import prepare_swell, read_swell
from temporal_features import add_session_dynamics, add_temporal_features

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "swell_kw_behavioral.tab"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
TARGET = "workload_state"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("SWELL-KW data is missing. Run: python scripts/setup_swell_kw.py")
    raw = read_swell(DATA_PATH)
    df, target, group_col = prepare_swell(raw)
    df = add_temporal_features(df, window=5, group_col=group_col)
    df = add_session_dynamics(df)
    feature_columns = [c for c in df.columns if c not in {target, group_col} and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_columns].replace([float("inf"), float("-inf")], pd.NA).apply(pd.to_numeric, errors="coerce")
    y = df[target].astype(str)
    groups = df[group_col].astype(str) if group_col else pd.Series(range(len(df)), index=df.index)
    return X, y, groups, feature_columns, group_col


def build_models():
    return {
        "random_forest_baseline": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=350, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced"))]),
        "adaptive_temporal_gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(learning_rate=0.055, max_iter=350, max_leaf_nodes=17, l2_regularization=1.2, random_state=42))]),
        "rbf_svm_comparator": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42))]),
    }


def evaluate():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    X, y, groups, feature_columns, group_col = load_data()
    # Hold out complete participants. This prevents temporal rows from the same
    # participant appearing in both train and test, a common source of leakage.
    unique_groups = groups.unique()
    train_groups, test_groups = train_test_split(unique_groups, test_size=0.20, random_state=42)
    train_mask = groups.isin(train_groups)
    test_mask = groups.isin(test_groups)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    g_train = groups.loc[train_mask]

    models = build_models()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, model in models.items():
        cv_f1 = cross_val_score(model, X_train, y_train, groups=g_train, cv=cv, scoring="f1_macro")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        rows.append({"model": name, "cv_macro_f1_mean": float(cv_f1.mean()), "cv_macro_f1_std": float(cv_f1.std()), "test_accuracy": float(accuracy_score(y_test, pred)), "test_macro_precision": float(precision), "test_macro_recall": float(recall), "test_macro_f1": float(f1), "train_participants": int(len(train_groups)), "test_participants": int(len(test_groups))})
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(cm); ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(RESULTS_DIR / f"confusion_matrix_{name}.png", dpi=160); plt.close(fig)
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    joblib.dump(fitted[best_name], MODELS_DIR / "best_model.joblib")
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.joblib")
    metadata = {
        "system": "Adaptive Temporal Behavioral Fusion",
        "dataset": "SWELL-KW",
        "dataset_doi": "10.17026/DANS-X55-69ZP",
        "target": TARGET,
        "target_semantics": "experimental workload condition; interpreted as a cognitive-load proxy, not a clinical diagnosis",
        "best_model": best_name,
        "selection_metric": "5-fold stratified group cross-validation macro F1",
        "participant_holdout": True,
        "features": feature_columns,
        "n_samples": int(len(X)),
        "classes": sorted(y.unique().tolist()),
        "group_column": group_col,
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False)); print(f"Selected model: {best_name}")


if __name__ == "__main__":
    evaluate()
