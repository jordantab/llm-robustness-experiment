import os
import json

def get_structure(obj, prefix=''):
    """Extract structure of JSON object showing types and sample values"""
    structure = {}
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                structure[key] = get_structure(value)
            else:
                # Store type and a sample value
                structure[key] = f"{type(value).__name__} (e.g., {str(value)[:50]}...)" if value else type(value).__name__
    
    elif isinstance(obj, list) and obj:
        # Show structure of first item as sample
        structure = f"list of {len(obj)} items, first item sample: {get_structure(obj[0])}"
        
    return structure

def analyze_json_files(directory_path):
    output = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    output.append(f"File: {file_path}")
                    output.append("Structure:")
                    structure = get_structure(data)
                    output.append(json.dumps(structure, indent=2))
                    output.append("-" * 80)
                    
                except json.JSONDecodeError:
                    output.append(f"Error: Could not parse JSON in {file_path}")
                except Exception as e:
                    output.append(f"Error reading {file_path}: {str(e)}")
    
    # Print to console instead of saving
    print('\n'.join(output))
    
    # Write output to file
    with open('json_structure.txt', 'w') as f:
        f.write('\n'.join(output))

directory_path = "./pb_results/progressnew"
analyze_json_files(directory_path)