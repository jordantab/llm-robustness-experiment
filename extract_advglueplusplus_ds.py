from datasets import load_dataset
from huggingface_hub import login

login("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")

def extract_and_store(task):
    # Load the dataset from Hugging Face
    dataset = load_dataset("AI-Secure/DecodingTrust", "adv-glue-plus-plus", split=task)

    if task == "qnli":
        aux_col = ["sentence", "question"]
    elif task == "mnli":
        aux_col = ["sentence", "question"]
    elif task == "qqp":
        aux_col = ["question2", "question1"]
    elif task == "sst2":
        aux_col = ["sentence"]

    # Group by attack type and select 60 instances from each group
    attack_column = "attack_type"  # Replace with the actual column name for attack type if it's different
    grouped = df.groupby(attack_column).head(60)

    # Filter the dataset to include only the required columns
    columns_to_include = ["idx", "label", "sentence", "question", attack_column]
    filtered_data = grouped[columns_to_include]

    # Define output file name
    output_file = f"advglueplusplus_{task_name}.csv"

    # Save to CSV
    filtered_data.to_csv(output_file, index=False)

    # Output file saved
    output_file