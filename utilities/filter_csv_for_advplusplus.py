import pandas as pd

# Load the two CSV files
csv1 = pd.read_csv('./benchmarks/reduced_prompts.csv')
csv2 = pd.read_csv('./benchmarks/advglueplusplus/formatted_prompts_with_originals.csv')

# Find rows where 'combined_prompt' matches in both CSVs
matched_rows = csv2[csv2['combined_prompt'].isin(csv1['combined_prompt'])]

# Save the result to a new CSV
matched_rows.to_csv('./benchmarks/final_advplusplus.csv', index=False)

print("Matched rows have been saved to '/benchmarks/final_advplusplus.csv'.")
