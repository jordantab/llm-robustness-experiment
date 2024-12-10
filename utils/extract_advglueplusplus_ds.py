from datasets import load_dataset
from huggingface_hub import login
import pandas as pd

login("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")

def extract_and_store(task):
    # Load the dataset from Hugging Face
    dataset = load_dataset("AI-Secure/DecodingTrust", "adv-glue-plus-plus", split=task)
    dataset = dataset.to_pandas()

    if task == "qnli":
        aux_col = ["sentence", "question"]
    elif task == "mnli":
        aux_col = ["hypothesis", "premise"]
    elif task == "qqp":
        aux_col = ["question2", "question1"]
    elif task == "sst2":
        aux_col = ["sentence"]

    # Group by attack type and select 60 instances from each group
    attack_types = ["semattack", "textbugger", "textfooler", "sememepso", "bertattack"]
    final_data = []
    for attack in attack_types: 
        filtered = dataset[dataset["method"] == attack].head(60)

        # Filter the dataset to include only the required columns
        columns_to_include = ["index", "label"] + aux_col + ["method"]
        filtered_data = filtered[columns_to_include]

        final_data.append(filtered_data)
    
    final_data = pd.concat(final_data, ignore_index=True)

    # Define output file name
    output_file = f"./benchmarks/advglueplusplus/advglueplusplus_{task}.csv"

    # Save to CSV
    final_data.to_csv(output_file, index=False)

    # Output file saved
    output_file

if __name__ == "__main__": 
    tasks = ["sst2", "qnli", "qqp", "mnli"]
    for task in tasks: 
        extract_and_store(task)
