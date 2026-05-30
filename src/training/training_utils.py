"""Training utilities for weighted loss and model artifact storage."""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import Trainer  # type: ignore


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
) -> None:
    """Save the current trainer model only if it beats the saved macro F1."""

    info_path = final_model_dir / "best_model_info.json"
    current_best_f1 = float(trainer.state.best_metric or 0.0)
    saved_best_f1 = _load_saved_best_f1(info_path)

    if saved_best_f1 is not None and current_best_f1 <= saved_best_f1:
        print(
            "Keeping existing final_model: "
            f"saved macro_f1={saved_best_f1:.6f}, "
            f"current macro_f1={current_best_f1:.6f}."
        )
        return

    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_model_dir))  # type: ignore
    tokenizer.save_pretrained(str(final_model_dir))
    _write_best_model_info(
        info_path=info_path,
        trainer=trainer,
        best_macro_f1=current_best_f1,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    print(f"Updated final_model with macro_f1={current_best_f1:.6f}.")


def _load_saved_best_f1(info_path: Path) -> float | None:
    if not info_path.exists():
        return None

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return float(json.load(f)["best_macro_f1"])
    except (KeyError, TypeError, ValueError, JSONDecodeError):
        return None


def _write_best_model_info(
    *,
    info_path: Path,
    trainer: Any,
    best_macro_f1: float,
    run_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    best_log = _find_best_log(trainer)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_macro_f1": best_macro_f1,
                "best_macro_f1_best_threshold": best_log.get(
                    "eval_macro_f1_best_threshold"
                ),
                "best_decision_threshold": best_log.get("eval_decision_threshold"),
                "metric_for_best_model": trainer.args.metric_for_best_model,
                "run_name": run_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "best_global_step": trainer.state.best_global_step,
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
