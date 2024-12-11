#!/usr/bin/env bash

# Specify your model ID here
MODEL_ID="llama2:13b"

# List of tasks to evaluate
TASKS=("mnli" "qnli" "qqp" "sst2")

# Path to your Python script
SCRIPT_PATH="advglueplusplus_eval_2.py"

# Optional: Activate your Python environment if needed
# source /path/to/venv/bin/activate

# Run each task
for TASK in "${TASKS[@]}"; do
    echo "Running evaluation for task: $TASK with model: $MODEL_ID"
    python $SCRIPT_PATH --model-id "$MODEL_ID" --task "$TASK"
    echo "Completed evaluation for task: $TASK"
    echo "----------------------------------------"
done

echo "All tasks completed."
