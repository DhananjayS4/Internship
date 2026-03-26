"""
Advanced Model Training for EASD
Trains an ensemble ML pipeline on WESAD features with:
- StandardScaler normalization
- RandomForest + GradientBoosting + VotingClassifier
- Stratified 5-fold cross-validation
- Saves best model + metrics JSON

Run from project root:
    python ml/train_advanced.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils import resample

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WESAD_CSV    = os.path.join(BASE_DIR, "data", "wesad_features.csv")
SYNTH_CSV    = os.path.join(BASE_DIR, "data", "synthetic_anxiety_dataset.csv")
MODEL_OUT    = os.path.join(BASE_DIR, "ml", "models", "anxiety_model_v2.joblib")
METRICS_OUT  = os.path.join(BASE_DIR, "data", "model_metrics.json")

FEATURE_COLS = [
    "emg_rms", "emg_var", "emg_mean",
    "acc_mean", "acc_std", "acc_max",
]
WESAD_EXTRA_COLS = ["hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "eda_mean"]


def load_data():
    if os.path.exists(WESAD_CSV):
        print(f"[OK] Loading WESAD features from: {WESAD_CSV}")
        df = pd.read_csv(WESAD_CSV)
        # Determine available feature columns
        available_feats = [c for c in FEATURE_COLS + WESAD_EXTRA_COLS if c in df.columns]
        df = df[available_feats + ["label"]].dropna()
        source = "WESAD"
    else:
        print(f"[!] WESAD CSV not found. Falling back to: {SYNTH_CSV}")
        df = pd.read_csv(SYNTH_CSV)
        available_feats = [c for c in FEATURE_COLS if c in df.columns]
        df = df[available_feats + ["label"]].dropna()
        source = "synthetic"

    return df, available_feats, source


def balance_classes(df):
    """SMOTE-style: upsample minority class to match majority."""
    majority = df[df["label"] == df["label"].value_counts().idxmax()]
    minority = df[df["label"] != df["label"].value_counts().idxmax()]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    return pd.concat([majority, minority_up]).sample(frac=1, random_state=42)


def build_candidates():
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    voting = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft"
    )
    return {
        "RandomForest":       rf,
        "GradientBoosting":   gb,
        "VotingEnsemble":     voting,
    }


def evaluate(name, clf, X, y, cv):
    print(f"\n  [{name}] Cross-validating ...")
    results = cross_validate(
        clf, X, y,
        cv=cv,
        scoring=["accuracy", "f1"],
        return_train_score=True,
        n_jobs=-1
    )
    test_acc = results["test_accuracy"].mean()
    test_f1  = results["test_f1"].mean()
    print(f"    Accuracy : {test_acc:.4f}  F1 : {test_f1:.4f}")
    return test_acc, test_f1


def train():
    print("=" * 60)
    print("  EASD Advanced Model Training")
    print("=" * 60)

    df, feat_cols, source = load_data()
    print(f"\nDataset: {source.upper()}  |  {len(df)} samples  |  "
          f"Features: {len(feat_cols)}")
    print(f"Class distribution:\n{df['label'].value_counts().to_string()}\n")

    df = balance_classes(df)
    X = df[feat_cols].values
    y = df["label"].values

    # Scaler inside pipeline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = build_candidates()

    best_name, best_score, best_f1 = None, -1, -1
    scores = {}

    print("Running 5-fold cross-validation on all candidates:")
    for name, clf in candidates.items():
        acc, f1 = evaluate(name, clf, X_scaled, y, cv)
        scores[name] = {"accuracy": round(acc, 4), "f1": round(f1, 4)}
        if acc > best_score:
            best_score, best_f1, best_name = acc, f1, name

    print(f"\n[OK] Best model: {best_name}  (Accuracy={best_score:.4f}, F1={best_f1:.4f})")

    # Final fit on all data
    best_clf = candidates[best_name]
    best_clf.fit(X_scaled, y)

    # Wrap scaler + model into a single pipeline object for inference
    pipe = Pipeline([("scaler", scaler), ("model", best_clf)])

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    dump(pipe, MODEL_OUT)
    print(f"[OK] Model saved -> {MODEL_OUT}")

    # Save metrics
    metrics = {
        "model_name":    best_name,
        "data_source":   source,
        "n_samples":     int(len(df)),
        "feature_cols":  feat_cols,
        "cv_scores":     scores,
        "best_accuracy": round(best_score, 4),
        "best_f1":       round(best_f1, 4),
    }
    with open(METRICS_OUT, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"[OK] Metrics saved -> {METRICS_OUT}")
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
