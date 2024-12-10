import json 
import os
import glob
import pandas as pd
from datetime import datetime
import numpy as np

def calculate_metrics(results):
    """Calculate precision, recall, f1 for binary classification"""
    if not results:
        return 0, 0, 0, 0
    
    tp = fp = tn = fn = 0
    for result in results:
        pred = result.get('prediction')
        label = result.get('label')
        if pred is None or label is None:
            continue
        
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return accuracy, precision, recall, f1

def parse_progress_files(directory):
    """Parse all JSON files in the progress folder"""
    results = []
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        model_id = data.get('model_id', '')
        dataset = data.get('dataset', '')
        attack_name = data.get('attack_name', '')
        total_samples = data.get('total_samples', 0)
        
        # Get latest timestamp
        last_update = data.get('last_update', '')
        if last_update:
            try:
                timestamp = datetime.strptime(last_update, "%Y%m%d_%H%M%S")
            except:
                timestamp = None
        else:
            timestamp = None
            
        prompt_results = data.get('prompt_results', {})
        num_prompts = len(prompt_results)
        
        # Calculate metrics across all prompts
        all_results = []
        total_completed = 0
        successful_perturbations = 0
        
        # First prompt is always the original unperturbed prompt
        original_prompt = next(iter(prompt_results))
        if original_prompt in prompt_results:
            original_accuracy = prompt_results[original_prompt].get('final_accuracy', 0)
        else:
            original_accuracy = 0
            
        # Calculate success rate for perturbed prompts
        perturbed_accuracies = []
        for prompt, prompt_data in list(prompt_results.items())[1:]:  # Skip first prompt
            accuracy = prompt_data.get('final_accuracy', 0)
            if isinstance(accuracy, (int, float)):
                perturbed_accuracies.append(accuracy)
                
        # Calculate PDR as drop in accuracy
        avg_perturbed_accuracy = np.mean(perturbed_accuracies) if perturbed_accuracies else 0
        pdr = max(0, min(1, original_accuracy - avg_perturbed_accuracy))
            
        for prompt, prompt_data in prompt_results.items():
            results_list = prompt_data.get('results', [])
            if isinstance(results_list, list):
                all_results.extend(results_list)
                total_completed += prompt_data.get('completed_samples', 0)
        
        accuracy, precision, recall, f1 = calculate_metrics(all_results)
        
        results.append({
            'timestamp': timestamp,
            'model_id': model_id,
            'task': f"{dataset}_{attack_name}",
            'dataset': dataset,
            'attack_name': attack_name,
            'num_samples': total_samples,
            'num_prompts': num_prompts,
            'num_samples_total': total_completed,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pdr': pdr,
            'original_accuracy': original_accuracy,
            'perturbed_accuracy': avg_perturbed_accuracy,
            'file_name': os.path.basename(file_path)
        })
    
    return results

def save_to_csv(results, output_file):
    """Save results to CSV file"""
    df = pd.DataFrame(results)
    
    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    # Format timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    # Directory containing the progress JSON files
    progress_dir = "./pb_results/progressnew/"
    
    # Parse files and get results
    results = parse_progress_files(progress_dir)
    
    # Save to CSV
    output_file = "progress_results.csv"
    save_to_csv(results, output_file)

if __name__ == "__main__":
    main()