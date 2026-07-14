"""
Training model for multi-label classification
"""

import argparse
import datetime
import os
from typing import Any, cast

from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
)

from src.settings import Settings
from src.training.dataset import get_tokenized_dataset
from src.training.metrics import compute_metrics, compute_metrics_at_fixed_threshold
from src.training.training_utils import (
    WeightedTrainer,
    compute_pos_weight,
    get_best_decision_threshold,
    get_best_validation_metrics,
    has_validation_improved,
    save_best_model_if_improved,
)


def run_training(
    epochs: int = Settings.DEFAULT_TRAINING_EPOCHS,
    batch_size: int = Settings.DEFAULT_TRAINING_BATCH_SIZE,
    learning_rate: float = Settings.DEFAULT_LEARNING_RATE,
    dry_run: bool = False,
    skip_test: bool = False,
) -> None:
    """Configures and runs the model training.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training and evaluation.
        learning_rate: Initial learning rate for training.
        dry_run: If True, runs a quick test with a small subset of the data.
        skip_test: If True, do not evaluate the held-out test split in this run.
    """

    print("Tokenize data...")
    tokenized_dataset, tokenizer = get_tokenized_dataset()

    if dry_run:
        print("Dry Run: Reduce size for testing-purpose...")
        tokenized_dataset["train"] = cast(Any, tokenized_dataset["train"]).select(
            range(Settings.DRY_RUN_TRAIN_SIZE)
        )
        tokenized_dataset["validation"] = cast(
            Any, tokenized_dataset["validation"]
        ).select(range(Settings.DRY_RUN_VALIDATION_SIZE))
        tokenized_dataset["test"] = cast(Any, tokenized_dataset["test"]).select(
            range(Settings.DRY_RUN_TEST_SIZE)
        )
        epochs = 1

    print(f"Init model: {Settings.MODEL_NAME}")
    num_labels = len(Settings.create_movie_tag_list())  # type: ignore
    model: Any = cast(Any, AutoModelForSequenceClassification).from_pretrained(
        Settings.MODEL_NAME,
        problem_type="multi_label_classification",
        num_labels=num_labels,
    )

    print("Config training-arguments...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}-epochs{epochs}_bs{batch_size}_lr{learning_rate:g}"
    logging_dir = Settings.MODELS_DIR / "logs" / run_name

    os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)

    training_args = TrainingArguments(
        output_dir=str(Settings.MODELS_DIR / "checkpoints"),
        logging_dir=str(logging_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=Settings.WEIGHT_DECAY,
        warmup_steps=Settings.WARMUP_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model=Settings.METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        seed=Settings.RANDOM_SEED,
        data_seed=Settings.RANDOM_SEED,
        save_total_limit=Settings.SAVE_TOTAL_LIMIT,
        use_cpu=False,  # Allow MPS or CUDA if available
        report_to=Settings.TRAINING_REPORT_TO,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    pos_weight = compute_pos_weight(tokenized_dataset["train"])

    print("init Trainer...")
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
    )

    print("start Training...")
    trainer.train()  # type: ignore
    print(f"Best validation metrics: {get_best_validation_metrics(trainer)}")

    if dry_run:
        if not skip_test:
            _evaluate_held_out_test_split(trainer, tokenized_dataset["test"])
        print("Dry run completed, model not saved.")
        return

    final_model_dir = Settings.MODELS_DIR / "final_model"
    if not has_validation_improved(trainer=trainer, final_model_dir=final_model_dir):
        save_best_model_if_improved(
            trainer=trainer,
            tokenizer=tokenizer,
            final_model_dir=final_model_dir,
            run_name=run_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        print(
            "Held-out test split was not evaluated because validation did not improve."
        )
        return

    test_metrics = None
    if skip_test:
        print("Held-out test split was not evaluated because --skip_test is set.")
    else:
        test_metrics = _evaluate_held_out_test_split(trainer, tokenized_dataset["test"])

    save_best_model_if_improved(
        trainer=trainer,
        tokenizer=tokenizer,
        final_model_dir=final_model_dir,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        test_metrics=test_metrics,
    )

    print("Training completed.")


def _evaluate_held_out_test_split(trainer: Any, test_dataset: Any) -> dict[str, float]:
    validation_threshold = get_best_decision_threshold(
        trainer, Settings.DECISION_THRESHOLD
    )
    print(
        "evaluate best validation checkpoint on held-out test split "
        f"with validation threshold={validation_threshold:.3f}..."
    )

    original_compute_metrics = trainer.compute_metrics
    trainer.compute_metrics = compute_metrics_at_fixed_threshold(validation_threshold)
    try:
        test_metrics = trainer.evaluate(
            eval_dataset=test_dataset,
            metric_key_prefix="test",
        )
    finally:
        trainer.compute_metrics = original_compute_metrics

    print(f"Held-out test metrics: {test_metrics}")
    return cast(dict[str, float], test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train structural movie-tag predictor model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=Settings.DEFAULT_TRAINING_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=Settings.DEFAULT_TRAINING_BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=Settings.DEFAULT_LEARNING_RATE,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Run a quick test with minimal data"
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help=(
            "Skip held-out test evaluation. Use this for hyperparameter experiments "
            "and run src.training.evaluate_final_model once afterwards."
        ),
    )

    args = parser.parse_args()
    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dry_run=args.dry_run,
        skip_test=args.skip_test,
    )
