import os
import json
import glob
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_metrics(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def main():
    # Directory where checkpoints are stored
    checkpoint_dir = "checkpoints_baseline"  # adjust if needed
    
    # Pattern to match progress files for adversarial attacks
    # These files might look like: adv_attack_{model_id}_{dataset_name}_{attack_name}_progress.json
    progress_files = glob.glob(os.path.join(checkpoint_dir, "adv_attack_*_progress.json"))
    
    # Data structure to hold metrics for each attack
    # Each attack may have multiple prompts, so we will aggregate all predictions and labels together.
    attack_data = {}

    for pf in progress_files:
        with open(pf, 'r') as f:
            data = json.load(f)
        
        # Extract the attack_name from the file or from the JSON structure
        # The JSON stored might have: "attack_name" in top-level keys
        # If not, we infer it from the filename
        attack_name = data.get("attack_name", None)
        if attack_name is None:
            # File name format: adv_attack_{model_id}_{dataset_name}_{attack_name}_progress.json
            # We can parse from filename
            filename = os.path.basename(pf)
            # filename split by underscore: ["adv","attack","{model_id}","{dataset_name}","{attack_name}_progress.json"]
            parts = filename.split("_")
            # Last meaningful part before "progress.json" should be the attack_name
            # For safety, we rejoin all parts after "adv_attack_{model_id}_{dataset_name}"
            # The structure is known from original code: adv_attack_model_dataset_attackname_progress.json
            if len(parts) >= 5:
                # parts[0] = adv
                # parts[1] = attack
                # parts[2] = model_id
                # parts[3] = dataset_name
                # The rest (except the last which includes _progress.json) is attack name
                # Actually, we expect attack_name to be a single token. If not, adjust accordingly.
                attack_name = "_".join(parts[4:]).replace("_progress.json", "")
            else:
                attack_name = "unknown_attack"
        
        prompt_results = data.get("prompt_results", {})
        
        all_labels = []
        all_preds = []
        
        for prompt, presult in prompt_results.items():
            results = presult.get("results", [])
            # Extract labels and predictions
            for r in results:
                if "label" in r and "prediction" in r:
                    all_labels.append(r["label"])
                    all_preds.append(r["prediction"])
        
        # Store or aggregate results for this attack
        # If the script was run multiple times and we have duplicates, we could aggregate them here,
        # but typically one file corresponds to one attack and includes all prompts.
        if len(all_labels) > 0 and len(all_preds) > 0:
            # Calculate metrics for this attack
            accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_preds)
            num_samples = len(all_labels)
            
            # If attack_name already in attack_data, we might combine metrics. 
            # But since each file presumably contains all prompts for that attack, we just store once.
            # If you need to average multiple runs of the same attack, you'd store them in a list and average later.
            attack_data[attack_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "num_samples": num_samples
            }
        else:
            # No samples? Skip or set them to None/0
            attack_data[attack_name] = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "num_samples": 0
            }

    # If you have multiple files per attack (e.g., partial runs), you could adjust this section to average metrics
    # For now, we assume one final progress file per attack that includes all results.

    # Write out final result file
    output_file = os.path.join(checkpoint_dir, "final_attack_metrics.tsv")
    with open(output_file, 'w') as fout:
        # Header
        fout.write("attack_name\taccuracy\tprecision\trecall\tf1\tnum_samples\n")
        for attack_name, metrics in attack_data.items():
            fout.write(f"{attack_name}\t{metrics['accuracy']}\t{metrics['precision']}\t{metrics['recall']}\t{metrics['f1']}\t{metrics['num_samples']}\n")

    print(f"Final aggregated metrics written to {output_file}")

if __name__ == "__main__":
    main()
