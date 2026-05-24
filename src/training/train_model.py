"""
Training model for multi-label classification
"""

import argparse
import datetime
import os
from typing import Any, cast

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score  # type: ignore
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from src.settings import Settings
from src.training.dataset import get_tokenized_dataset


def compute_metrics(
    eval_pred: EvalPrediction | tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> dict[str, float]:
    """Calculates macro F1-score and ROC-AUC for multi-label classification.

    Args:
        eval_pred: Either an EvalPrediction object or a tuple of (logits, labels).

    Returns:
        A dictionary with 'macro_f1' and 'roc_auc' scores.
    """

    logits: Any
    labels: Any
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits, labels = eval_pred.predictions, eval_pred.label_ids

    # Sigmoid function
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs >= Settings.DECISION_THRESHOLD).astype(float)

    # Calculate metrics
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(labels, probs, average="macro", multi_class="ovr")
    except Exception as e:
        print(f"Warnung bei ROC-AUC Berechnung (z.B. fehlende Klassenvarianz): {e}")
        roc_auc = 0.0

    return {"macro_f1": float(macro_f1), "roc_auc": float(roc_auc)}


def run_training(epochs: int = 12, batch_size: int = 24, dry_run: bool = False) -> None:
    """Configures and runs the model training.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training and evaluation.
        dry_run: If True, runs a quick test with a small subset of the data.
    """

    print("Tokenize data...")
    tokenized_dataset, tokenizer = get_tokenized_dataset()

    if dry_run:
        print("Dry Run: Reduce size for testing-purpose...")
        tokenized_dataset["train"] = cast(Any, tokenized_dataset["train"]).select(
            range(20)
        )
        tokenized_dataset["test"] = cast(Any, tokenized_dataset["test"]).select(
            range(10)
        )
        epochs = 1

    print(f"Init model: {Settings.MODEL_NAME}")
    num_labels = len(Settings.create_mood_list())  # type: ignore
    model: Any = cast(Any, AutoModelForSequenceClassification).from_pretrained(
        Settings.MODEL_NAME,
        problem_type="multi_label_classification",
        num_labels=num_labels,
    )

    print("Config training-arguments...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}-epochs{epochs}_bs{batch_size}"
    logging_dir = Settings.MODELS_DIR / "logs" / run_name

    os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)

    training_args = TrainingArguments(
        output_dir=str(Settings.MODELS_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=2,
        use_cpu=False,  # Allow MPS or CUDA if available
        report_to="tensorboard",
    )

    print("init Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("start Training...")
    trainer.train()  # type: ignore

    print(f"Safe best model in: {Settings.MODELS_DIR}")
    model.save_pretrained(str(Settings.MODELS_DIR / "final_model"))  # type: ignore
    tokenizer.save_pretrained(str(Settings.MODELS_DIR / "final_model"))
    print("Training completed, model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train structural movie mood predictor model"
    )
    parser.add_argument(
        "--epochs", type=int, default=12, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument(
        "--dry_run", action="store_true", help="Run a quick test with minimal data"
    )

    args = parser.parse_args()
    run_training(epochs=args.epochs, batch_size=args.batch_size, dry_run=args.dry_run)
