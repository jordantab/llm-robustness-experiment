#!/bin/bash

# Define models and benchmarks
MODELS=("mixtral" "llama2")
BENCHMARKS=("promptbench" "advglue++")
DATASETS=("sst2" "qnli" "qqp" "mnli")

# Create a logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/evaluation_${TIMESTAMP}.log"

echo "Starting evaluation at $(date)" | tee -a "$LOG_FILE"

# Loop through all combinations
for model in "${MODELS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            echo "Running evaluation for:" | tee -a "$LOG_FILE"
            echo "- Model: $model" | tee -a "$LOG_FILE"
            echo "- Benchmark: $benchmark" | tee -a "$LOG_FILE"
            echo "- Dataset: $dataset" | tee -a "$LOG_FILE"
            
            # Run the evaluation script
            python adv_ahp_eval.py \
                --model_id "${model}:7b" \
                --benchmark "$benchmark" \
                --dataset_name "$dataset" 2>&1 | tee -a "$LOG_FILE"
            
            # Add separator for readability
            echo "----------------------------------------" | tee -a "$LOG_FILE"
            
            # Optional: Add small delay between runs to prevent potential rate limiting
            sleep 2
        done
    done
done

echo "Evaluation completed at $(date)" | tee -a "$LOG_FILE"