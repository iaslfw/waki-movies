#!/bin/bash

EXPERIMENTS=(
    "10 8 2e-5"
    "12 8 2e-5"
    "8 8 3e-5"
    "10 8 3e-5"
    "8 4 2e-5"
)

echo "Starting selected hyperparameter tuning..."

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    read -r EPOCHS BATCH_SIZE LEARNING_RATE <<< "$EXPERIMENT"

    echo "====================================================="
    echo "Starting new Run: Epochs = $EPOCHS | Batch-Size = $BATCH_SIZE | Learning-Rate = $LEARNING_RATE"
    echo "====================================================="

    uv run --no-sync python -m src.training.train_model \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE"

    echo "Run done!"
    echo ""
done

echo "Finished selected experiments!"
