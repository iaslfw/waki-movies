"""Metric helpers for multi-label movie mood training."""

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score  # type: ignore
from transformers import EvalPrediction  # type: ignore

from src.settings import Settings


def find_best_threshold(
    labels: Any, probs: np.ndarray[Any, Any]
) -> tuple[float, float]:
    """Find the best global threshold for macro F1 on the eval split."""

    best_threshold = Settings.DECISION_THRESHOLD
    best_macro_f1 = 0.0

    for threshold in np.arange(
        Settings.THRESHOLD_SEARCH_START,
        Settings.THRESHOLD_SEARCH_STOP,
        Settings.THRESHOLD_SEARCH_STEP,
    ):
        predictions = (probs >= threshold).astype(float)
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = float(macro_f1)
            best_threshold = float(threshold)

    return best_macro_f1, best_threshold


def compute_metrics(
    eval_pred: EvalPrediction | tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> dict[str, float]:
    """Calculate macro F1 and ROC-AUC for multi-label classification."""

    logits: Any
    labels: Any
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits, labels = eval_pred.predictions, eval_pred.label_ids

    probs = 1 / (1 + np.exp(-logits))

    macro_f1_best_threshold, decision_threshold = find_best_threshold(labels, probs)
    threshold_predictions = (probs >= Settings.DECISION_THRESHOLD).astype(float)
    macro_f1 = f1_score(labels, threshold_predictions, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(labels, probs, average="macro", multi_class="ovr")
    except Exception as e:
        print(f"Warnung bei ROC-AUC Berechnung (z.B. fehlende Klassenvarianz): {e}")
        roc_auc = 0.0

    return {
        "macro_f1": float(macro_f1),
        "macro_f1_best_threshold": float(macro_f1_best_threshold),
        "decision_threshold": float(decision_threshold),
        "roc_auc": float(roc_auc),
    }
