import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import json
from typing import Dict

# Set the file path and other configurations here
FILE_PATH = "checkpoints/detailed_results_adv_llama2_advglue++.csv"  # Path to the detailed results CSV file
OUTPUT_CSV = "results/ahp/adv_advglue++_llama2_metrics_by_attack.csv"  # File to save the aggregated metrics in CSV format
MODEL_NAME = "llama2"  # Model name
BENCHMARK = "advglue++"  # Benchmark name
GROUP_BY = "method"  # Column to group by: 'dataset' or 'attack_name'

def calculate_grouped_metrics(data: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """
    Calculate metrics grouped by a specific column (e.g., 'dataset', 'attack_name').

    Args:
        data (pd.DataFrame): Data containing 'label' and 'prediction' columns along with group-by column.
        group_by (str): Column name to group by.

    Returns:
        pd.DataFrame: DataFrame containing grouped metrics.
    """
    metrics_list = []
    
    for group, group_data in data.groupby(group_by):
        true_labels = group_data['label']
        predictions = group_data['prediction']
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        metrics_list.append({
            'model': MODEL_NAME,
            'benchmark': BENCHMARK,
            'dataset': group_data['dataset'].iloc[0] if 'dataset' in group_data.columns else None,
            'attack_name': group,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(group_data)
        })
    
    return pd.DataFrame(metrics_list)

def main():
    # Check if the file exists
    if not os.path.exists(FILE_PATH):
        print(f"Error: File {FILE_PATH} does not exist.")
        return
    
    # Load data
    try:
        data = pd.read_csv(FILE_PATH)
    except Exception as e:
        print(f"Error loading file {FILE_PATH}: {str(e)}")
        return
    
    # Ensure required columns exist
    required_columns = ['label', 'prediction', GROUP_BY]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {', '.join(missing_columns)}")
        return

    # Calculate grouped metrics
    grouped_metrics = calculate_grouped_metrics(data, GROUP_BY)

    # Save results to CSV
    grouped_metrics.to_csv(OUTPUT_CSV, index=False)
    print(f"Aggregated metrics saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
