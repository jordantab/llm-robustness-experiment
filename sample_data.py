import os
import pandas as pd

# Specify the arguments here
INPUT_FILE = "benchmarks/promptattack.csv"
OUTPUT_FILE = "benchmarks/promptattack_reduced.csv"
GROUP_BY = "attack_name"  # Column to group by (e.g., "dataset" or "attack_name")
REDUCTION_FACTOR = 0.5  # Proportion of samples to retain (e.g., 0.5 for 50%)

def reduce_samples(input_file: str, output_file: str, group_by: str, reduction_factor: float = 0.5):
    """
    Reduce the number of samples uniformly across a specified column and labels within that column.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the reduced dataset.
        group_by (str): Column name to group by for uniform sampling.
        reduction_factor (float): Proportion of samples to retain (default: 0.5).
    """
    # Load the dataset
    df = pd.read_csv(input_file)

    # Ensure the reduction factor is valid
    if not (0 < reduction_factor <= 1):
        raise ValueError("Reduction factor must be between 0 and 1.")

    # Calculate target samples
    total_samples = len(df)
    target_samples = int(total_samples * reduction_factor)

    # Group by the specified column
    grouped = df.groupby(group_by)

    reduced_dfs = []
    for group_name, group_df in grouped:
        # Calculate target samples per label within each group
        label_counts = group_df["label"].value_counts()
        target_per_label = target_samples // (len(grouped) * len(label_counts))

        for label, count in label_counts.items():
            label_df = group_df[group_df["label"] == label]
            sampled_df = label_df.sample(n=min(target_per_label, len(label_df)), random_state=42)
            reduced_dfs.append(sampled_df)

    # Combine the sampled data
    reduced_df = pd.concat(reduced_dfs, ignore_index=True)

    # Save the reduced dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    reduced_df.to_csv(output_file, index=False)
    print(f"Reduced dataset saved to {output_file}. Total samples: {len(reduced_df)}")

if __name__ == "__main__":
    # Run the function with the specified arguments
    reduce_samples(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        group_by=GROUP_BY,
        reduction_factor=REDUCTION_FACTOR
    )
