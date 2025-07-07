#!/bin/bash

# Batch evaluation script
# Usage: ./batch_eval.sh [checkpoint_dir]

CHECKPOINT_DIR=${1:-"checkpoints"}

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Directory '$CHECKPOINT_DIR' not found"
    exit 1
fi

CHECKPOINT_FILES=$(find "$CHECKPOINT_DIR" -name "*.pt" -type f)

if [ -z "$CHECKPOINT_FILES" ]; then
    echo "No best_model*.pt files found in '$CHECKPOINT_DIR'"
    exit 1
fi

echo "Found $(echo "$CHECKPOINT_FILES" | wc -l) checkpoint files"

while IFS= read -r checkpoint_file; do
    filename=$(basename "$checkpoint_file" .pt)

    echo "Processing: $checkpoint_file"

    python eval.py -cn eval_voxel \
        "checkpoint='$checkpoint_file'" \
        "name=rgb"

done <<< "$CHECKPOINT_FILES"

echo "Done"
