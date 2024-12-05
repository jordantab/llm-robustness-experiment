import pandas as pd
# Specify the path to your text file
file_path = 'fk_text_pred.txt'

# Read the file into a list
with open(file_path, 'r') as file:
    lines = file.read().splitlines()  # Read lines and remove the newline character

# Print the list
print(lines)

def load_dataset_for_benchmark(benchmark): 
    if benchmark == "promptbench":
        path = './benchmarks/promptattack.csv'
        data = pd.read_csv(path)

        # Extract the `combined_prompt` and `label` columns
        extracted_data = data[["combined_prompt", "label", "idx"]]
        return extracted_data
    elif benchmark == "flipkart":
        path = './benchmarks/flipkart-sentiment.csv'
        df = pd.read_csv(path)
        filtered_df = df[df['Summary'].notna()]
        length_filtered_df = filtered_df[filtered_df['Summary'].str.len().between(
            150, 160)]
        length_filtered_df = length_filtered_df[:300]
        # Extract the `Summary` and `Sentiment` columns
        extracted_data = length_filtered_df[["Summary", "Sentiment"]]
        return extracted_data
    else: 
        print("Not a valid benchmark")

ds = load_dataset_for_benchmark("flipkart")

res = pd.DataFrame(columns=["pred", "true_label"])
for i in range(len(ds)):
    instance = ds.iloc[i]
    new_row = {"pred": lines[i], "true_label": instance["Sentiment"]}

