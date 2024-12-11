import json
import csv
import glob
import os

def extract_prompts_from_file(file_path):
    print(f"Processing file: {file_path}")  # Debug print
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Skip checklist and stresstest
    attack_name = data.get('attack_name', '')
    dataset = data.get('dataset', '')

    print(f"Found attack_name in file: {attack_name}")  # Debug print

    if attack_name in ['checklist', 'stresstest']:
        return []

    # Extract all prompts
    prompts = list(data.get('prompt_results', {}).keys())
    print(f"Found {len(prompts)} prompts in file")  # Debug print

    return [(dataset, attack_name, prompt) for prompt in prompts]

def main():
    # Get all JSON files
    json_files = glob.glob('checkpoints_baseline/*_progress.json')
    print(f"Found {len(json_files)} JSON files:")  # Debug print
    for f in json_files:
        print(f"  {f}")  # Debug print

    # Collect all entries
    csv_data = []
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            attack_name = data.get('attack_name', '').lower()

        print(f"\nProcessing {file_path}")  # Debug print
        print(f"Attack name: {attack_name}")  # Debug print

        # Skip if attack not defined in the file
        if not attack_name:
            print("Skipping: attack name not found in file")  # Debug print
            continue

        # Extract prompts
        new_entries = extract_prompts_from_file(file_path)
        csv_data.extend(new_entries)

        print(f"Collected {len(new_entries)} prompts from {file_path}")  # Debug print

    # Sort by attack_name
    csv_data = sorted(csv_data, key=lambda x: x[1])

    # Write to CSV
    with open('pb_prompts_all.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'attack_name', 'prompt'])
        writer.writerows(csv_data)

    print("\nExtraction completed. Prompts written to pb_prompts.csv.")

if __name__ == "__main__":
    main()
