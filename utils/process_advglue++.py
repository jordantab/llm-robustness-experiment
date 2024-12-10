import pandas as pd
import ast
import numpy as np

# List your 4 CSV files here
files = [
    "results/advglue++_sst2_evaluation_results.csv",
    "results/advglue++_qnli_evaluation_results.csv",
    "results/advglue++_qqp_evaluation_results.csv",
    "results/advglue++_mnli_evaluation_results.csv"
]

# Read and concatenate all results
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# Function to parse list columns
def parse_list_column(col):
    return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Parse precision, recall, f1 columns from strings to lists
df['precision'] = parse_list_column(df['precision'])
df['recall'] = parse_list_column(df['recall'])
df['f1'] = parse_list_column(df['f1'])

def macro_average(lst):
    """Compute the macro average of a list of per-class metrics."""
    return np.mean(lst) if lst else np.nan

# Group by task
grouped = df.groupby('task', as_index=False).agg({
    'model_id': 'first',
    'num_samples': 'mean',
    'accuracy': 'mean',
    'asr': 'mean',
    # We'll handle precision, recall, f1 separately
})

# For each task, compute the macro-averaged precision/recall/f1
task_rows = []
for _, row in grouped.iterrows():
    task = row['task']
    # Filter df for this task
    task_df = df[df['task'] == task]

    # It might be that there are multiple rows for one task, so average them first:
    # First, gather all precision/recall/f1 arrays for the task
    all_precisions = task_df['precision'].tolist()
    all_recalls = task_df['recall'].tolist()
    all_f1s = task_df['f1'].tolist()

    # Average them element-wise if multiple rows per task
    # Convert to numpy arrays. If there's only one row, this will still work.
    # Check if there's more than one row:
    if len(all_precisions) > 1:
        # Ensure they all have the same length
        length_p = len(all_precisions[0])
        if not all(len(p) == length_p for p in all_precisions):
            raise ValueError(f"Not all precision arrays have the same length for task {task}.")
        precision_arr = np.array(all_precisions)
        recall_arr = np.array(all_recalls)
        f1_arr = np.array(all_f1s)

        avg_precision_list = precision_arr.mean(axis=0).tolist()
        avg_recall_list = recall_arr.mean(axis=0).tolist()
        avg_f1_list = f1_arr.mean(axis=0).tolist()
    else:
        # Only one row, just use it directly
        avg_precision_list = all_precisions[0]
        avg_recall_list = all_recalls[0]
        avg_f1_list = all_f1s[0]

    # Now compute macro averages for the task
    task_macro_precision = macro_average(avg_precision_list)
    task_macro_recall = macro_average(avg_recall_list)
    task_macro_f1 = macro_average(avg_f1_list)

    # Update the row with macro-averaged values
    new_row = row.to_dict()
    new_row['precision'] = task_macro_precision
    new_row['recall'] = task_macro_recall
    new_row['f1'] = task_macro_f1
    task_rows.append(new_row)

final_tasks_df = pd.DataFrame(task_rows)

# Compute the overall macro average across tasks (now all metrics are scalar)
macro_accuracy = final_tasks_df['accuracy'].mean()
macro_asr = final_tasks_df['asr'].mean()
macro_precision = final_tasks_df['precision'].mean()
macro_recall = final_tasks_df['recall'].mean()
macro_f1 = final_tasks_df['f1'].mean()

# Add the combined row
combined_row = {
    'task': 'combined',
    'model_id': 'combined',
    'num_samples': final_tasks_df['num_samples'].mean(),
    'accuracy': macro_accuracy,
    'asr': macro_asr,
    'precision': macro_precision,
    'recall': macro_recall,
    'f1': macro_f1
}

final_df = pd.concat([final_tasks_df, pd.DataFrame([combined_row])], ignore_index=True)

# Save to CSV
final_df.to_csv("results/baseline/llama2:13b/advglue++_results.csv", index=False)
print("Results saved to results/baseline/llama2:13b/advglue++_results.csv")
