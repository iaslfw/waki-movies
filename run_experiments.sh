#!/bin/bash

EPOCHS_LIST=(5 8 10 12)
BATCH_SIZE_LIST=(8 16 24 32)

echo "Starting Hyperparameter-Tuning..."

for EPOCHS in "${EPOCHS_LIST[@]}"; do
    for BATCH in "${BATCH_SIZE_LIST[@]}"; do
        
        echo "====================================================="
        echo "Starting new Run: Epochs = $EPOCHS | Batch-Size = $BATCH"
        echo "====================================================="
        
        uv run --no-sync python -m src.training.train_model --epochs $EPOCHS --batch_size $BATCH
        
        echo "Run done!"
        echo ""
        
    done
done

echo "Finished all experiements!"