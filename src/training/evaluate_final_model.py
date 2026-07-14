"""Evaluate the saved best validation model on the held-out test split."""

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
from src.training.metrics import compute_metrics_at_fixed_threshold
from src.training.training_utils import (
    WeightedTrainer,
    compute_pos_weight,
    load_best_model_info,
    update_best_model_test_metrics,
)


def evaluate_final_model(
    batch_size: int = Settings.DEFAULT_TRAINING_BATCH_SIZE,
) -> dict[str, float]:
    """Evaluate the current final_model exactly once against the held-out test split."""

    final_model_dir = Settings.MODELS_DIR / "final_model"
    model_info = load_best_model_info(final_model_dir)
    if model_info.get("metric_for_best_model") != Settings.METRIC_FOR_BEST_MODEL:
        raise ValueError(
            "Saved final_model metadata does not match the current selection metric."
        )

    validation_threshold = float(
        model_info.get("best_decision_threshold") or Settings.DECISION_THRESHOLD
    )
    print(
        "Evaluating saved final_model on held-out test split "
        f"with validation threshold={validation_threshold:.3f}..."
    )

    print(f"Load model: {final_model_dir}")
    model: Any = cast(Any, AutoModelForSequenceClassification).from_pretrained(
        str(final_model_dir)
    )

    print("Tokenize data...")
    tokenized_dataset, tokenizer = get_tokenized_dataset(
        tokenizer_name_or_path=final_model_dir
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"final_test_{timestamp}"
    logging_dir = Settings.MODELS_DIR / "logs" / run_name

    os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)

    training_args = TrainingArguments(
        output_dir=str(Settings.MODELS_DIR / "test_evaluation"),
        logging_dir=str(logging_dir),
        per_device_eval_batch_size=batch_size,
        seed=Settings.RANDOM_SEED,
        data_seed=Settings.RANDOM_SEED,
        use_cpu=False,
        report_to=Settings.TRAINING_REPORT_TO,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    pos_weight = compute_pos_weight(tokenized_dataset["train"])
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_at_fixed_threshold(validation_threshold),
        pos_weight=pos_weight,
    )

    test_metrics = cast(
        dict[str, float],
        trainer.evaluate(
            eval_dataset=tokenized_dataset["test"],
            metric_key_prefix="test",
        ),
    )
    update_best_model_test_metrics(
        final_model_dir=final_model_dir,
        test_metrics=test_metrics,
        test_run_name=run_name,
    )

    print(f"Held-out test metrics: {test_metrics}")
    print(f"TensorBoard log run: {run_name}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate saved final_model on the held-out test split"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=Settings.DEFAULT_TRAINING_BATCH_SIZE,
        help="Batch size for final evaluation",
    )
    args = parser.parse_args()
    evaluate_final_model(batch_size=args.batch_size)
