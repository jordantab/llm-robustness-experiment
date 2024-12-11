import pandas as pd
import re

# Read the input files
prompts_df = pd.read_csv('pb_prompts_all.csv')
sst2_df = pd.read_csv('sst2_sample.csv')

# Initialize empty lists to store the combined data
combined_data = []

# Iterate through each prompt
for _, prompt_row in prompts_df.iterrows():
    # Only process rows for sst2 dataset
    if prompt_row['dataset'] != 'sst2':
        continue
    
    # Iterate through each SST2 sample
    for _, sst2_row in sst2_df.iterrows():
        # Check if prompt contains placeholder, if not just append the sentence
        if '{' in prompt_row['prompt']:
            combined_prompt = re.sub(r'\{content[^\}]*\}', sst2_row['sentence'], prompt_row['prompt'])
        else:
            combined_prompt = prompt_row['prompt'] + ' ' + sst2_row['sentence']
        
        # Create a new row with all required fields
        combined_row = {
            'dataset': 'sst2',
            'attack_name': prompt_row['attack_name'],
            'combined_prompt': combined_prompt,
            'prompt': prompt_row['prompt'],
            'sentence': sst2_row['sentence'],
            'label': sst2_row['label'],
            'idx': sst2_row['idx']
        }
        
        combined_data.append(combined_row)

# Create DataFrame from combined data
result_df = pd.DataFrame(combined_data)

# Add sampling logic
sampling_counts = {
    'textbugger': 100,
    'deepwordbug': 100,
    'bertattack': 100,
    'semantic': 200,
    'textfooler': 100
}

# Sample rows for each attack with balanced labels
sampled_dfs = []
for attack, count in sampling_counts.items():
    attack_df = result_df[result_df['attack_name'] == attack]
    # Calculate samples needed per label (half of total count)
    samples_per_label = count // 2
    
    # Sample equally from each label
    positive_samples = attack_df[attack_df['label'] == 1].sample(n=samples_per_label, random_state=42)
    negative_samples = attack_df[attack_df['label'] == 0].sample(n=samples_per_label, random_state=42)
    
    # Combine positive and negative samples
    balanced_sample = pd.concat([positive_samples, negative_samples])
    sampled_dfs.append(balanced_sample)

# Combine all sampled DataFrames
result_df = pd.concat(sampled_dfs, ignore_index=True)

# Save to CSV
result_df.to_csv('promptattack.csv', index=False)