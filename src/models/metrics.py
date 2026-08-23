# src/metrics.py
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """Calculates the ratio of negative to positive cases for scale_pos_weight."""
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    return float(neg_count / pos_count) if pos_count > 0 else 1.0


def evaluate_thresholds(
    y_true: pd.Series,
    y_probas: np.ndarray,
    thresholds: List[float] | None = None,
) -> pd.DataFrame:
    """Evaluates precision, recall, and F1-score across various probability thresholds."""
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.277, 0.30, 0.35, 0.40, 0.45, 0.50]

    results = []
    for thresh in thresholds:
        preds = (y_probas >= thresh).astype(int)
        results.append(
            {
                "Threshold": thresh,
                "Precision": precision_score(y_true, preds, pos_label=1, zero_division=0),
                "Recall": recall_score(y_true, preds, pos_label=1, zero_division=0),
                "F1_Score": f1_score(y_true, preds, pos_label=1, zero_division=0),
            }
        )

    df_results = pd.DataFrame(results)
    print("\n--- Threshold Tuning Results ---")
    print(df_results.to_string(index=False))
    return df_results


def evaluate_model_performance(
    y_true: pd.Series, y_pred: np.ndarray, digits: int = 3
) -> str:
    """Generates and prints the classification report."""
    report = classification_report(y_true, y_pred, digits=digits)
    print("\n--- Classification Report ---")
    print(report)
    return report