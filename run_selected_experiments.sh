#!/bin/bash

set -euo pipefail

EXPERIMENTS=(
    # Main search around the strongest previous region.
    "8 8 3e-5"
    "10 8 3e-5"
    "12 8 3e-5"
    "6 8 3e-5"

    # Intermediate learning rate between the previous 2e-5 and 3e-5 runs.
    "8 8 2.5e-5"
    "10 8 2.5e-5"
    "12 8 2.5e-5"
    "6 8 2.5e-5"

    # Conservative learning rate with longer schedules.
    "8 8 2e-5"
    "10 8 2e-5"
    "12 8 2e-5"
    "14 8 2e-5"
    "6 8 2e-5"

    # Faster learning rate probes; these should show quickly whether 4e-5 is too hot.
    "4 8 4e-5"
    "6 8 4e-5"
    "8 8 4e-5"

    # Slow learning rate probes; only useful with longer schedules.
    "12 8 1.5e-5"
    "16 8 1.5e-5"

    # Batch-size probes kept narrow because previous broad searches favored batch_size=8.
    "8 4 2e-5"
    "8 4 3e-5"
    "8 16 3e-5"
)

echo "Starting selected hyperparameter tuning (${#EXPERIMENTS[@]} training runs)..."

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    read -r EPOCHS BATCH_SIZE LEARNING_RATE <<< "$EXPERIMENT"

    echo "====================================================="
    echo "Starting new Run: Epochs = $EPOCHS | Batch-Size = $BATCH_SIZE | Learning-Rate = $LEARNING_RATE"
    echo "====================================================="

    uv run --no-sync python -m src.training.train_model \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --skip_test

    echo "Run done!"
    echo ""
done

echo "Evaluating selected best model on held-out test split..."
uv run --no-sync python -m src.training.evaluate_final_model --batch_size 8

echo "Finished selected experiments and final test evaluation!"
