import json


def read_json(file_path):
    """Read a JSON file and return the contents as a list of values."""
    with open(file_path, 'r') as file:
        return json.load(file)


def compare_jsons(file1_path, file2_path):
    """Compare values at each index in two JSON files and print mismatches."""
    # Read the contents of both files
    file1_data = read_json(file1_path)
    file2_data = read_json(file2_path)

    # Check if both files have the same number of elements
    if len(file1_data) != len(file2_data):
        print("Files have different lengths.")
        return

    # Compare values at each index
    mismatches = []
    for i in range(len(file1_data)):
        if file1_data[i] != file2_data[i]:
            mismatches.append((i, file1_data[i], file2_data[i]))

    # Print out the mismatches
    if mismatches:
        print("Found mismatches at the following indices:")
        print(len(mismatches))
        for index, val1, val2 in mismatches:
            print(f"Index {index}: File1='{val1}' | File2='{val2}'")
    else:
        print("No mismatches found. Files are identical.")

# Replace 'file1.csv' and 'file2.csv' with the paths to your CSV files
compare_jsons('predictions/baseline/flipkart/llama2:7b2_predictions.json', 'predictions/baseline/flipkart/llama2:7b_predictions.json')
