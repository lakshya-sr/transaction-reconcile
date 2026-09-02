#!/usr/bin/env python3
"""
Train the Cluster-Level Residual XGBoost Classifier.

Optimizes for PR-AUC on hard negatives and exports model and calibrated
probability threshold artifacts.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ai.features import FEATURE_COLUMNS

DATA_PATH = ROOT_DIR / "data" / "raw" / "train_features.csv"
ARTIFACT_DIR = ROOT_DIR / "src" / "ai" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_gw_bank.json"
THRESHOLD_PATH = ARTIFACT_DIR / "xgb_gw_bank_threshold.json"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"

MIN_PRECISION_TARGET = 0.98


def select_calibrated_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    min_precision: float = MIN_PRECISION_TARGET,
) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.50

    precisions = precisions[:-1]
    recalls = recalls[:-1]

    candidates = []
    for idx, t in enumerate(thresholds):
        p = float(precisions[idx])
        r = float(recalls[idx])
        if p >= min_precision:
            f1 = 2 * (p * r) / (p + r + 1e-9)
            candidates.append((f1, float(t), p, r))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
        return candidates[0][1]

    # Fallback: Pick highest precision threshold
    best_p_idx = int(np.argmax(precisions))
    return float(thresholds[best_p_idx])


def train_and_export():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at {DATA_PATH}. Run dataset_builder first.")

    df = pd.read_csv(DATA_PATH)
    print(f"[*] Loaded {len(df)} samples from {DATA_PATH}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    scale_pos = max(1.0, float(neg_count) / max(1, pos_count))

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=350,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=3,
        gamma=1.0,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
    )

    print("[*] Training XGBoost classifier...")
    model.fit(X_train, y_train)

    # Evaluate
    test_probs = model.predict_proba(X_test)[:, 1]
    ap_score = average_precision_score(y_test, test_probs)
    calibrated_threshold = select_calibrated_threshold(y_test, test_probs, min_precision=MIN_PRECISION_TARGET)

    # Threshold performance
    y_pred_calibrated = (test_probs >= calibrated_threshold).astype(int)
    print(f"[✔] PR-AUC (Average Precision): {ap_score:.4f}")
    print(f"[✔] Calibrated Decision Threshold: {calibrated_threshold:.4f}")
    print("\n--- Test Set Classification Report ---")
    print(classification_report(y_test, y_pred_calibrated, digits=4))

    # Export Artifacts
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    threshold_payload = {
        "threshold": round(float(calibrated_threshold), 4),
        "min_precision_target": MIN_PRECISION_TARGET,
        "pr_auc": round(float(ap_score), 4),
        "feature_columns": FEATURE_COLUMNS,
    }
    THRESHOLD_PATH.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")
    SCHEMA_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")

    print(f"[✔] Exported Model: {MODEL_PATH}")
    print(f"[✔] Exported Threshold Config: {THRESHOLD_PATH}")


def main():
    train_and_export()


if __name__ == "__main__":
    main()
