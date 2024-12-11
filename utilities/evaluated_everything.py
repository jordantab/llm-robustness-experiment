import pandas as pd 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_flipkart(path, model): 
    df = pd.read_csv(path) 

    rows_removed  = 0
    if model == "mixtral:8x7b" or model == "llama2:13b":
        rows_to_remove = df[df["pred"] == "mixed"]
        rows_removed = len(rows_to_remove)
        df = df[df["pred"] != "mixed"]  # Keep rows where 'pred' is not 'mixed'
    
        # Reset the index after filtering
        df.reset_index(drop=True, inplace=True)

    preds = df["pred"]
    labels = df["true_label"]
    # Define the valid set of values
    valid_values = {"neutral", "positive", "negative"}

    # Find invalid entries in preds
    invalid_preds = df[~df["pred"].isin(valid_values)]

    # Find invalid entries in labels
    invalid_labels = df[~df["true_label"].isin(valid_values)]

    # Print invalid entries
    if not invalid_preds.empty:
        print("Invalid entries in 'pred':")
        print(invalid_preds)

    if not invalid_labels.empty:
        print("Invalid entries in 'true_label':")
        print(invalid_labels)


    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rows_removed': rows_removed
    }
    return metrics

def evaluate_advplusplus_sst2(path,model):
    df = pd.read_csv(path)

    rows_removed  = 0
    if model == "mixtral:8x7b" or model == "llama2:7b":
        # Identify rows to remove
        print(df.iloc[2]["pred"])
        rows_to_remove = df[df["pred"] == 'Error']
        print(rows_to_remove)

        # Count rows removed
        rows_removed = len(rows_to_remove)

        # Filter out rows where 'pred' is 'Error'
        df = df[df["pred"] != 'Error']

        # Reset the index
        df.reset_index(drop=True, inplace=True)

    df.loc[df["pred"] == "positive", "pred"] = 1

    # Assign 0 to rows where 'pred' is "negative"
    df.loc[df["pred"] == "negative", "pred"] = 0
    preds = df["pred"].to_list()
    labels = df["label"].to_list()
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rows_removed': rows_removed
    }
    return metrics

def evaluate_advplusplus_task(path,task):
    df = pd.read_csv(path)
    if task == "sst2":
        rows = [0, 74]
    elif task == "qnli":
        rows = [75,148]
    elif task == "mnli":
        rows = [149,214]
    elif task == "qqp":
        rows = [215,289]
    
    df = df.iloc[rows[0]:rows[1] + 1]
    print(len(df))
    rows_removed  = 0
    # Identify rows to remove
    rows_to_remove = df[df["pred"] == 'Error']
    print(rows_to_remove)

    # Count rows removed
    rows_removed = len(rows_to_remove)

    # Filter out rows where 'pred' is 'Error'
    df = df[df["pred"] != 'Error']

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    df['pred'] = pd.to_numeric(df['pred'])
    preds = df['pred'].to_list()
    string_instances = [item for item in preds if isinstance(item, str)]

    if string_instances:
        print("Found string instances:")
        for string_item in string_instances:
            print(string_item)
    else:
        print("No string instances found.")
    labels = df["label"].to_list()
    accuracy = accuracy_score(labels, preds)
    if task == "mnli": 
        strategy = "weighted"
    else:
        strategy = "binary"
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=strategy)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rows_removed': rows_removed
    }
    return metrics

def evaluate_advplusplus_method(path,method):
    df = pd.read_csv(path)
    # get rows where the values in "method" column equals the method variable
    df = df[df["method"] == method]

    rows_removed  = 0
    # Identify rows to remove
    rows_to_remove = df[df["pred"] == 'Error']
    print(rows_to_remove)

    # Count rows removed
    rows_removed = len(rows_to_remove)

    # Filter out rows where 'pred' is 'Error'
    df = df[df["pred"] != 'Error']

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    preds = df["num_pred"].to_list()
    labels = df["label"].to_list()
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rows_removed': rows_removed
    }
    return metrics


def eval_promptbench(pred_path, attack):
    df = pd.read_csv(pred_path)

    if attack == "bertattack":
        rows = [0, 59]
    elif attack == "deepwordbug":
        rows = [60,119]
    elif attack == "semantic":
        rows = [120,179]
    elif attack == "textbugger":
        rows = [180,239]
    elif attack == "textfooler":
        rows = [240,299]
    
    df = df.iloc[rows[0]:rows[1] + 1]
    print(len(df))
    rows_removed  = 0
    # Identify rows to remove
    rows_to_remove = df[df["pred"] == 'Error']
    print(rows_to_remove)

    # Count rows removed
    rows_removed = len(rows_to_remove)

    # Filter out rows where 'pred' is 'Error'
    df = df[df["pred"] != 'Error']

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    df.loc[df["pred"] == "positive", "pred"] = 1

    # Assign 0 to rows where 'pred' is "negative"
    df.loc[df["pred"] == "negative", "pred"] = 0
    preds = df["pred"].to_list()
    labels = df["true_label"].to_list()
    df['pred'] = pd.to_numeric(df['pred'])
    preds = df['pred'].to_list()
    labels = df["true_label"].to_list()
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rows_removed': rows_removed
    }
    return metrics







# print(evaluate_flipkart("./predictions/mixtral8x7b_predictions_flipkart_ICR.csv", "mixtral:8x7b"))
# print(evaluate_flipkart("./predictions/llama27b_predictions_flipkart_ICR.csv", "llama2:7b"))
# print(evaluate_flipkart("./predictions/llama213b_predictions_flipkart_ICR.csv", "llama2:13b"))
# # print(evaluate_advplusplus_sst2("llama27b_predictions_advplusplus_sst2.csv", "llama2:7b"))
# print("llama sst2", evaluate_advplusplus_task("llama27b_plusplus_ICR.csv", "sst2"))
# print("llama mnli", evaluate_advplusplus_task("llama27b_plusplus_ICR.csv", "mnli"))
# print("llama qnli", evaluate_advplusplus_task("llama27b_plusplus_ICR.csv", "qnli"))
# print("llama qqp", evaluate_advplusplus_task("llama27b_plusplus_ICR.csv", "qqp"))
# print("mixtral sst2", evaluate_advplusplus_task("mixtral_plusplus_ICR.csv", "sst2"))
# print("mixtral mnli", evaluate_advplusplus_task("mixtral_plusplus_ICR.csv", "mnli"))
# print("mixtral qnli", evaluate_advplusplus_task("mixtral_plusplus_ICR.csv", "qnli"))
# print("mixtral qqp", evaluate_advplusplus_task("mixtral_plusplus_ICR.csv", "qqp"))
# print("llama bertattack", evaluate_advplusplus_method("llama27b_plusplus_ICR.csv", "bertattack"))
# print("llama sememepso", evaluate_advplusplus_method("llama27b_plusplus_ICR.csv", "sememepso"))
# print("llama semattack", evaluate_advplusplus_method("llama27b_plusplus_ICR.csv", "semattack"))
# print("llama textbugger", evaluate_advplusplus_method("llama27b_plusplus_ICR.csv", "textbugger"))
# print("llama textfooler", evaluate_advplusplus_method("llama27b_plusplus_ICR.csv", "textfooler"))
# print("mixtral bertattack", evaluate_advplusplus_method("mixtral_plusplus_ICR.csv", "bertattack"))
# print("mixtral sememepso", evaluate_advplusplus_method("mixtral_plusplus_ICR.csv", "sememepso"))
# print("mixtral semattack", evaluate_advplusplus_method("mixtral_plusplus_ICR.csv", "semattack"))
# print("mixtral textbugger", evaluate_advplusplus_method("mixtral_plusplus_ICR.csv", "textbugger"))
# print("mixtral textfooler", evaluate_advplusplus_method("mixtral_plusplus_ICR.csv", "textfooler"))
# print("llama13 sst2", evaluate_advplusplus_task("./predictions/llama2:13b_predictions_advplusplus_ICR.csv", "sst2"))
# print("llama13 mnli", evaluate_advplusplus_task("./predictions/llama2:13b_predictions_advplusplus_ICR.csv", "mnli"))
# print("llama13 qnli", evaluate_advplusplus_task("./predictions/llama2:13b_predictions_advplusplus_ICR.csv", "qnli"))
# print("llama13 qqp", evaluate_advplusplus_task("./predictions/llama2:13b_predictions_advplusplus_ICR.csv", "qqp"))
print("llama13 bert", eval_promptbench("./predictions/llama2:13b_predictions_promptbench_ICR.csv", "bertattack"))
print("llama13 deepbug", eval_promptbench("./predictions/llama2:13b_predictions_promptbench_ICR.csv", "deepwordbug"))
print("llama13 semantic", eval_promptbench("./predictions/llama2:13b_predictions_promptbench_ICR.csv", "semantic"))
print("llama13 textbugger", eval_promptbench("./predictions/llama2:13b_predictions_promptbench_ICR.csv", "textbugger"))
print("llama13 textfooler", eval_promptbench("./predictions/llama2:13b_predictions_promptbench_ICR.csv", "textfooler"))



