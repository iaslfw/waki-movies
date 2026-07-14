"""Metric helpers for multi-label movie-tag training."""

from collections.abc import Callable
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


def _unpack_eval_prediction(
    eval_pred: EvalPrediction | tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> tuple[Any, Any]:
    logits: Any
    labels: Any
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
    return logits, labels


def compute_multilabel_metrics(
    eval_pred: EvalPrediction | tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
    *,
    decision_threshold: float,
    search_best_threshold: bool,
) -> dict[str, float]:
    """Calculate macro F1 and ROC-AUC for multi-label classification."""

    logits, labels = _unpack_eval_prediction(eval_pred)

    probs = 1 / (1 + np.exp(-logits))

    threshold_predictions = (probs >= decision_threshold).astype(float)
    macro_f1 = f1_score(labels, threshold_predictions, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(labels, probs, average="macro", multi_class="ovr")
    except Exception as e:
        print(f"Warnung bei ROC-AUC Berechnung (z.B. fehlende Klassenvarianz): {e}")
        roc_auc = 0.0

    metrics = {
        "macro_f1": float(macro_f1),
        "decision_threshold": float(decision_threshold),
        "roc_auc": float(roc_auc),
    }

    if search_best_threshold:
        macro_f1_best_threshold, best_decision_threshold = find_best_threshold(
            labels, probs
        )
        metrics.update({
            "macro_f1_best_threshold": float(macro_f1_best_threshold),
            "decision_threshold": float(best_decision_threshold),
        })

    return metrics


def compute_metrics(
    eval_pred: EvalPrediction | tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> dict[str, float]:
    """Calculate validation metrics and search the best threshold on validation."""

    return compute_multilabel_metrics(
        eval_pred,
        decision_threshold=Settings.DECISION_THRESHOLD,
        search_best_threshold=True,
    )


def compute_metrics_at_fixed_threshold(
    decision_threshold: float,
) -> Callable[
    [EvalPrediction | tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
    dict[str, float],
]:
    """Build a metric function that evaluates without tuning on the given split."""

    def _compute_metrics(
        eval_pred: EvalPrediction | tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
    ) -> dict[str, float]:
        return compute_multilabel_metrics(
            eval_pred,
            decision_threshold=decision_threshold,
            search_best_threshold=False,
        )

    return _compute_metrics
