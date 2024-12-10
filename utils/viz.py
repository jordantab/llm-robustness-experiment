import json
import os
import csv
from tabulate import tabulate

def process_results_folder(folder_path):
    results = []
    
    # Read all json files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)
                
                # Handle different JSON structures
                if 'all_attack_results' in data:
                    # Handle final results files
                    for attack_name, attack_result in data['all_attack_results'].items():
                        results.append({
                            'model': data['model_id'],
                            'attack': attack_name,
                            'original_score': attack_result['original score'],
                            'attacked_score': attack_result['attacked score'],
                            'PDR': attack_result['PDR']
                        })
                elif 'result' in data and isinstance(data['result'], dict) and 'original score' in data['result']:
                    # Handle individual attack result files
                    results.append({
                        'model': data['model_id'],
                        'attack': data['attack_name'],
                        'original_score': data['result']['original score'],
                        'attacked_score': data['result']['attacked score'],
                        'PDR': data['result']['PDR']
                    })

    # Sort results by model and attack name
    results.sort(key=lambda x: (x['model'], x['attack']))
    
    # Prepare headers and data
    headers = ['Model', 'Attack', 'Original Score', 'Attacked Score', 'PDR']
    table_data = [[r['model'], r['attack'], r['original_score'], r['attacked_score'], r['PDR']] for r in results]
    
    # Generate and save table as txt
    table = tabulate(table_data, headers=headers, tablefmt='grid')
    with open('results_table.txt', 'w') as f:
        f.write(table)
    
    # Save as CSV
    with open('results_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_data)
    
    return table, table_data

# Example usage
folder_path = './pb_results/results'  # Replace with your actual folder path
table, data = process_results_folder(folder_path)
print("Table format:")
print(table)
print("\nCSV file has been created at: results_table.csv")