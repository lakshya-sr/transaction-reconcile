#!/usr/bin/env python3
"""Train the residual XGBoost classifier and export model + threshold artifacts."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
DATA_PATH = ROOT_DIR / "data" / "raw" / "train_features.csv"
ARTIFACT_DIR = ROOT_DIR / "src" / "ai" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_gw_bank.json"
THRESHOLD_PATH = ARTIFACT_DIR / "xgb_gw_bank_threshold.json"
FEATURE_COLUMNS = ["amount_diff", "time_delta_hours", "utr_fuzz_ratio"]
MIN_PRECISION_FOR_THRESHOLD = 0.92
AMOUNT_MATCH_TOLERANCE = 0.01


def select_best_threshold(y_true: pd.Series, probabilities: np.ndarray, min_precision: float = MIN_PRECISION_FOR_THRESHOLD) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    reward_scores = []

    for idx, threshold in enumerate(thresholds):
        p = float(precision[idx])
        r = float(recall[idx])
        if p < min_precision:
            continue
        # Precision-first scoring: reward accuracy more than recall.
        reward = (p ** 2) - 0.25 * (1.0 - r)
        reward_scores.append((reward, float(threshold), p, r))

    if not reward_scores:
        best_index = int(np.argmax(precision * recall))
        return float(thresholds[best_index])

    reward_scores.sort(key=lambda item: (item[0], item[2], item[3]), reverse=True)
    _, selected_threshold, _, _ = reward_scores[0]
    return float(selected_threshold)


def main():
    data = pd.read_csv(DATA_PATH)
    if data.empty:
        raise ValueError(f"Training data not found or empty: {DATA_PATH}")

    X = data[FEATURE_COLUMNS]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        max_depth=2,
        learning_rate=0.03,
        n_estimators=400,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.65,
        reg_lambda=3.0,
        gamma=2.5,
        scale_pos_weight=max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = select_best_threshold(y_test, probabilities)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}, indent=2), encoding="utf-8")

    print(f"[✔] XGBoost model saved to: {MODEL_PATH}")
    print(f"[✔] Threshold saved to: {THRESHOLD_PATH}")
    print(f"Optimal F1 probability threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
