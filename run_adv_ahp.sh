#!/bin/bash

# Define the models you want to test
models=("llama2:13b") 

# Define the robustness types
robustness_types=("adv")
num_samples=100

# Loop over each model
for model in "${models[@]}"; do

  # Set num_samples based on the model
  if [[ "$model" == "llama2:7b" ]]; then
    num_samples=1000
  elif [[ "$model" == "mixtral:8x7b" ]]; then
    num_samples=500
  fi

  # Loop over each robustness type
  for robustness_type in "${robustness_types[@]}"; do
    if [ "$robustness_type" == "adv" ]; then
      # For 'adv' robustness type
      benchmarks=("promptbench")
      for benchmark in "${benchmarks[@]}"; do
        echo "Running evaluation for Model: $model, Benchmark: $benchmark, Robustness Type: $robustness_type"
        python adv_ahp_eval.py --model_id "$model" --benchmark "$benchmark" --robustness_type "$robustness_type" --num_samples "$num_samples"
      done
    elif [ "$robustness_type" == "ood" ]; then
      # For 'OOD' robustness type
      benchmark="flipkart"
      dataset="flipkart"
      echo "Running evaluation for Model: $model, Benchmark: $benchmark, Dataset: $dataset, Robustness Type: $robustness_type"
      python ood_ahp_eval.py --model_id "$model" --benchmark "$benchmark" --dataset_name "$dataset" --robustness_type "$robustness_type"
    elif [ "$robustness_type" == "baseline" ]; then
      # For baseline mode, run without AHP steps
      benchmark="promptbench"  # You can choose a different benchmark if needed
      echo "Running baseline evaluation for Model: $model, Benchmark: $benchmark, Robustness Type: $robustness_type"
      python adv_ahp_eval.py --model_id "$model" --benchmark "$benchmark" --robustness_type "$robustness_type" --baseline --num_samples "$num_samples"
    fi
  done
done
