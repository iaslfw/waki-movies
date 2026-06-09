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
from src.training.metrics import compute_metrics
from src.training.training_utils import (
    WeightedTrainer,
    compute_pos_weight,
    save_best_model_if_improved,
)


def run_training(
    epochs: int = Settings.DEFAULT_TRAINING_EPOCHS,
    batch_size: int = Settings.DEFAULT_TRAINING_BATCH_SIZE,
    learning_rate: float = Settings.DEFAULT_LEARNING_RATE,
    dry_run: bool = False,
) -> None:
    """Configures and runs the model training.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training and evaluation.
        learning_rate: Initial learning rate for training.
        dry_run: If True, runs a quick test with a small subset of the data.
    """

    print("Tokenize data...")
    tokenized_dataset, tokenizer = get_tokenized_dataset()

    if dry_run:
        print("Dry Run: Reduce size for testing-purpose...")
        tokenized_dataset["train"] = cast(Any, tokenized_dataset["train"]).select(
            range(Settings.DRY_RUN_TRAIN_SIZE)
        )
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
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
    )

    print("start Training...")
    trainer.train()  # type: ignore

    if dry_run:
        print("Dry run completed, model not saved.")
        return

    save_best_model_if_improved(
        trainer=trainer,
        tokenizer=tokenizer,
        final_model_dir=Settings.MODELS_DIR / "final_model",
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    print("Training completed.")


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

    args = parser.parse_args()
    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dry_run=args.dry_run,
    )
