import json
import csv
import glob
import os

def extract_prompts_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    dataset = data.get('dataset', '')
    attack_name = data.get('attack_name', '').lower()  # Convert to lowercase immediately
    # Skip checklist and stresstest datasets
    if attack_name in ['checklist', 'stresstest']:
        print(f"Skipping file {file_path} with attack {attack_name}")  # Debug log
        return []
        
    prompts = list(data.get('prompt_results', {}).keys())
    
    print(f"Processing {file_path}: Found {len(prompts)} prompts from dataset {dataset} for attack {attack_name}")  # Debug log
    return [(dataset, attack_name, prompt) for prompt in prompts]

def main():
    # Get all JSON files in the pb_results/progress directory
    json_files = glob.glob('pb_results/progress/*_progress.json')
    print(f"Found {len(json_files)} JSON files")  # Debug log
    
    # Prepare data for CSV
    csv_data = []
    seen_prompts = set()  # Track unique prompts
    
    for file_path in json_files:
        for dataset, attack_name, prompt in extract_prompts_from_file(file_path):
            # Only add if we haven't seen this prompt before
            # if prompt not in seen_prompts:
            csv_data.append((dataset, attack_name, prompt))
                # seen_prompts.add(prompt)
    
    print(f"Total unique prompts: {len(seen_prompts)}")  # Debug log
    
    # Write to CSV
    with open('prompts.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['dataset', 'attack_name', 'prompt'])
        # Write data
        writer.writerows(csv_data)

if __name__ == "__main__":
    main()