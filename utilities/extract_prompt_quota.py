import json
import csv
import glob
import os

def extract_prompts_from_file(file_path, remaining_quota):
    print(f"Processing file: {file_path}")  # Debug print
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Skip checklist and stresstest
    attack_name = data.get('attack_name', '')
    dataset = data.get('dataset', '')
    
    print(f"Found attack_name in file: {attack_name}")  # Debug print
    
    if attack_name in ['checklist', 'stresstest']:
        return []
        
    # Take only the first N prompts based on remaining quota
    prompts = list(data.get('prompt_results', {}).keys())
    print(f"Found {len(prompts)} prompts in file")  # Debug print
    
    selected_prompts = prompts[:remaining_quota]
    return [(dataset, attack_name, prompt) for prompt in selected_prompts]

def main():
    # Define max prompts per attack
    max_prompts_dict = {
        'textfooler': 2,
        'bertattack': 2,
        'semantic': 4,
        'deepwordbug': 2,
        'textbugger': 2
    }
    
    # Keep track of collected prompts per attack
    collected_counts = {attack: 0 for attack in max_prompts_dict}
    
    # Get all JSON files
    json_files = glob.glob('checkpoints_baseline/*_progress.json')
    print(f"Found {len(json_files)} JSON files:")  # Debug print
    for f in json_files:
        print(f"  {f}")  # Debug print
    
    # Collect all entries
    csv_data = []
    for file_path in json_files:
        # Extract attack name from the JSON data instead of filename
        with open(file_path, 'r') as f:
            data = json.load(f)
            attack_name = data.get('attack_name', '').lower()
        
        print(f"\nProcessing {file_path}")  # Debug print
        print(f"Attack name: {attack_name}")  # Debug print
        
        # Skip if attack not in our target list or quota reached
        if attack_name not in max_prompts_dict:
            print(f"Skipping: attack {attack_name} not in target list")  # Debug print
            continue
            
        # Calculate remaining quota for this attack
        remaining_quota = max_prompts_dict[attack_name] - collected_counts[attack_name]
        if remaining_quota <= 0:
            print(f"Skipping: quota reached for {attack_name}")  # Debug print
            continue
            
        # Extract prompts
        new_entries = extract_prompts_from_file(file_path, remaining_quota)
        csv_data.extend(new_entries)
        
        # Update collected count
        collected_counts[attack_name] += len(new_entries)
        print(f"Collected {len(new_entries)} new prompts for {attack_name}")  # Debug print
    
    # Sort by attack_name
    csv_data = sorted(csv_data, key=lambda x: x[1])
    
    # Write to CSV
    with open('pb_prompts.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'attack_name', 'prompt'])
        writer.writerows(csv_data)
    
    # Print final statistics
    print("\nFinal collection counts:")
    for attack, count in collected_counts.items():
        print(f"{attack}: {count}/{max_prompts_dict[attack]}")

if __name__ == "__main__":
    main()