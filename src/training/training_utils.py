"""Training utilities for weighted loss and model artifact storage."""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import Trainer  # type: ignore

MODEL_SELECTION_SPLIT = "validation"
SPLIT_STRATEGY = "iterative_multilabel_stratified"


def compute_pos_weight(train_dataset: Any) -> torch.Tensor:
    labels = np.asarray(train_dataset["labels"], dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """Trainer using weighted BCE loss for multi-label classification."""

    def __init__(self, *args: Any, pos_weight: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **_: Any,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device)
        )
        loss = loss_fct(logits, labels.to(logits.dtype))
        return (loss, outputs) if return_outputs else loss


def save_best_model_if_improved(
    *,
    trainer: Any,
    tokenizer: Any,
    final_model_dir: Path,
    run_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    test_metrics: dict[str, float] | None = None,
) -> None:
    """Save the current trainer model only if it beats the saved validation metric."""

    info_path = final_model_dir / "best_model_info.json"
    metric_name = str(trainer.args.metric_for_best_model)
    current_best_metric = float(trainer.state.best_metric or 0.0)
    saved_best_metric = _load_saved_best_metric(info_path, metric_name)

    if not _has_validation_improved(current_best_metric, saved_best_metric):
        print(
            "Keeping existing final_model: "
            f"saved {metric_name}={saved_best_metric:.6f}, "
            f"current {metric_name}={current_best_metric:.6f}."
        )
        return

    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_model_dir))  # type: ignore
    tokenizer.save_pretrained(str(final_model_dir))
    _write_best_model_info(
        info_path=info_path,
        trainer=trainer,
        best_validation_metric=current_best_metric,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        test_metrics=test_metrics,
    )

    print(f"Updated final_model with {metric_name}={current_best_metric:.6f}.")


def has_validation_improved(*, trainer: Any, final_model_dir: Path) -> bool:
    """Return whether the current run beats the saved validation metric."""

    info_path = final_model_dir / "best_model_info.json"
    metric_name = str(trainer.args.metric_for_best_model)
    current_best_metric = float(trainer.state.best_metric or 0.0)
    saved_best_metric = _load_saved_best_metric(info_path, metric_name)
    return _has_validation_improved(current_best_metric, saved_best_metric)


def load_best_model_info(final_model_dir: Path) -> dict[str, Any]:
    """Load metadata for the saved final model and validate the split protocol."""

    info_path = final_model_dir / "best_model_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing best model metadata: {info_path}")

    with open(info_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    if model_info.get("selection_split") != MODEL_SELECTION_SPLIT:
        raise ValueError(
            "Saved final_model metadata does not match the current selection split."
        )
    if model_info.get("split_strategy") != SPLIT_STRATEGY:
        raise ValueError(
            "Saved final_model metadata does not match the current split strategy."
        )
    return model_info


def update_best_model_test_metrics(
    *,
    final_model_dir: Path,
    test_metrics: dict[str, float],
    test_run_name: str,
) -> None:
    """Write final held-out test metrics for the saved best validation model."""

    info_path = final_model_dir / "best_model_info.json"
    model_info = load_best_model_info(final_model_dir)
    model_info["test_split"] = "test"
    model_info["test_metrics"] = test_metrics
    model_info["test_evaluation_run_name"] = test_run_name

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)


def get_best_decision_threshold(trainer: Any, default_threshold: float) -> float:
    """Return the decision threshold selected on the best validation checkpoint."""

    best_log = _find_best_log(trainer)
    try:
        return float(best_log["eval_decision_threshold"])
    except (KeyError, TypeError, ValueError):
        return default_threshold


def get_best_validation_metrics(trainer: Any) -> dict[str, Any]:
    """Return the tracked metrics for the best validation checkpoint."""

    best_log = _find_best_log(trainer)
    metric_name = str(trainer.args.metric_for_best_model)
    return {
        "metric_for_best_model": metric_name,
        "best_validation_metric": float(trainer.state.best_metric or 0.0),
        "best_validation_macro_f1": best_log.get("eval_macro_f1"),
        "best_validation_macro_f1_best_threshold": best_log.get(
            "eval_macro_f1_best_threshold"
        ),
        "best_validation_roc_auc": best_log.get("eval_roc_auc"),
        "best_validation_loss": best_log.get("eval_loss"),
        "best_decision_threshold": best_log.get("eval_decision_threshold"),
        "best_global_step": trainer.state.best_global_step,
    }


def _load_saved_best_metric(info_path: Path, metric_name: str) -> float | None:
    if not info_path.exists():
        return None

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            if model_info.get("selection_split") != MODEL_SELECTION_SPLIT:
                return None
            if model_info.get("split_strategy") != SPLIT_STRATEGY:
                return None
            if model_info.get("metric_for_best_model") != metric_name:
                return None
            return float(model_info["best_validation_metric"])
    except (KeyError, TypeError, ValueError, JSONDecodeError):
        return None


def _has_validation_improved(
    current_best_metric: float, saved_best_metric: float | None
) -> bool:
    return saved_best_metric is None or current_best_metric > saved_best_metric


def _write_best_model_info(
    *,
    info_path: Path,
    trainer: Any,
    best_validation_metric: float,
    run_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    test_metrics: dict[str, float] | None,
) -> None:
    best_validation_metrics = get_best_validation_metrics(trainer)
    metric_name = str(trainer.args.metric_for_best_model)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_validation_metric": best_validation_metric,
                "best_validation_metric_name": metric_name,
                "best_validation_macro_f1": best_validation_metrics[
                    "best_validation_macro_f1"
                ],
                "best_validation_macro_f1_best_threshold": best_validation_metrics[
                    "best_validation_macro_f1_best_threshold"
                ],
                "best_validation_roc_auc": best_validation_metrics[
                    "best_validation_roc_auc"
                ],
                "best_validation_loss": best_validation_metrics[
                    "best_validation_loss"
                ],
                "best_decision_threshold": best_validation_metrics[
                    "best_decision_threshold"
                ],
                "selection_split": MODEL_SELECTION_SPLIT,
                "split_strategy": SPLIT_STRATEGY,
                "test_split": "test",
                "test_metrics": test_metrics,
                "metric_for_best_model": trainer.args.metric_for_best_model,
                "run_name": run_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "best_global_step": best_validation_metrics["best_global_step"],
            },
            f,
            indent=4,
        )


def _find_best_log(trainer: Any) -> dict[str, Any]:
    return next(
        (
            log
            for log in trainer.state.log_history
            if log.get("step") == trainer.state.best_global_step
            and "eval_decision_threshold" in log
        ),
        {},
    )
