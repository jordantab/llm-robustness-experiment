# import ollama

# response = ollama.generate(model='gemma:2b',
#                            prompt='what is qubit?')

# print(response['response'])\
import numpy as np
from typing import List, Tuple, Dict, Union
from langchain_ollama.llms import OllamaLLM
from datasets import load_dataset
from huggingface_hub import login
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from datetime import datetime
import pandas as pd
import re
from ICR import *
import argparse

login("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")

def calculate_asr(original_predictions: Union[List[int], np.ndarray],
                 adversarial_predictions: Union[List[int], np.ndarray],
                 true_labels: Union[List[int], np.ndarray]) -> float:
    """
    Calculate Attack Success Rate (ASR) according to the formula:
    ASR = Σ 1[f(A(x)) ≠ y] / 1[f(x) = y]
    
    Where:
    - f(A(x)) is the model's prediction on adversarial example (adversarial_predictions)
    - f(x) is the model's prediction on original example (original_predictions)
    - y is the ground truth label (true_labels)
    
    Args:
        original_predictions: Model predictions on original examples
        adversarial_predictions: Model predictions on adversarial examples
        true_labels: Ground truth labels
        
    Returns:
        float: Attack Success Rate
    """
    # Convert inputs to numpy arrays for vectorized operations
    original_predictions = np.array(original_predictions)
    adversarial_predictions = np.array(adversarial_predictions)
    true_labels = np.array(true_labels)
    
    # Find correctly classified original examples
    correctly_classified = (original_predictions == true_labels)
    
    if not np.any(correctly_classified):
        return 0.0  # Avoid division by zero if no correctly classified examples
    
    # Count successful attacks (misclassified adversarial examples)
    successful_attacks = (adversarial_predictions != true_labels)
    
    # Calculate ASR only for originally correct predictions
    asr = np.sum(successful_attacks[correctly_classified]) / np.sum(correctly_classified)
    
    return float(asr)

def parse_sentiment(response: str) -> int:
    """
    Parse the sentiment from LLM response
    
    Args:
        response: Raw response from LLM (e.g., "Sentiment: Positive")
        
    Returns:
        0 for negative, 1 for positive
    """
    # Convert to lowercase and remove extra whitespace
    response = response.lower().strip()
    
    # Fallback: check if 'positive' or 'negative' appears anywhere in response
    if 'positive' in response:
        return 'positive'
    elif 'negative' in response:
        return 'negative'
    
    # If no clear sentiment is found, return None to flag for investigation
    return None

def evaluate(model_id, task, dataset, task_to_keys, task_to_prompts, isICR=False):
    model = OllamaLLM(model=model_id)
    if task == 'sst2':
        cols = task_to_keys[task]
    elif task == "qnli":
        col_ques, col_sen = task_to_keys[task]
        col_org_ques = 'original_question'
        col_org_sentence = "original_sentence"
    elif task == "qqp":
        col_ques1, col_ques2 = task_to_keys[task]
        col_org_ques1 = 'original_question1'
        col_org_ques2 = 'original_question2'
    elif task == "mnli":
        col_premise, col_hypo = task_to_keys[task]
        col_org_premise = 'original_premise'
        col_org_hypo = 'original_hypothesis'

    processed = 0
    labels = dataset['label']
    print(labels)
    preds = []
    original_preds = []
    raw_responses = []  
    skipped = 0  
    for i in range(len(dataset)):
        if task == 'sst2':
            instance = dataset.iloc[i]
            sentence = instance[cols[0]]
            
            if isICR == True:
                # Rewrite the perturbed sentence with ICR
                rewritten_sentence = create_rewriting_prompt_plusplus(examples_per_attack, sentence, model_id)
                print(sentence)
                print(rewritten_sentence)
                prompt = task_to_prompts[task] + rewritten_sentence
            else: 
                print(sentence)
                prompt = task_to_prompts[task] + sentence
            
            response = model.invoke(prompt)
            raw_responses.append(response)
            
            # print(original_response, response)
            # Parse sentiment from response
            sentiment = parse_sentiment(response)
            print(response, sentiment, instance["label"])

            if sentiment is not None:
                # if sentiment == 0:
                #     print("NEGSATIVE")
                #     print(labels[i])
                preds.append(sentiment)
            else:
                print(f"Warning: Could not parse sentiment from response: {response}")
                # print(rewritten_sentence)
                skipped += 1
                preds.append('Error')  # Default to negative for unparseable responses
        
        if task == "qnli":
            sentence = instance[col_sen]
            question = instance[col_ques]
            if instance[col_org_ques] != '': 
                original_question = instance[col_org_ques]
                if isICR == True:
                    question, sentence = create_multiinput_rewriting_prompt(examples_per_attack, question, sentence, model_id)
                prompt = f"Does the sentence \"{sentence}\" answer the question \"{question}\"? The answer should only be exactly \"yes\" or \"no\". One word only, Nothing else."
                original_prompt = f"Does the sentence \"{sentence}\" answer the question \"{original_question}\"? The answer should only be exactly \"yes\" or \"no\". One word only, Nothing else."
            elif instance[col_org_sentence] != '': 
                original_sentence = instance[col_org_sentence]
                if isICR == True:
                    question, sentence = create_multiinput_rewriting_prompt(examples_per_attack, question, sentence, model_id)
                prompt = f"Does the sentence \"{sentence}\" answer the question \"{question}\"? The answer should only be exactly \"yes\" or \"no\". One word only, Nothing else."
                original_prompt = f"Does the sentence \"{original_sentence}\" answer the question \"{question}\"? The answer should only be exactly \"yes\" or \"no\". One word only, Nothing else."
            
            response = model.invoke(prompt)
            original_response = model.invoke(original_prompt)

            response = response.lower().strip()
            original_response = original_response.lower().strip()
            # print(response, original_response, labels[i])
            if 'yes' in response:
                preds.append(0)
            elif 'no' in response:
                preds.append(1)
            else: 
                print("did not get exact answer error; default to 0")
                preds.append(1)

            if 'yes' in original_response:
                original_preds.append(0)
            elif 'no' in original_response:
                original_preds.append(1)
            else: 
                print("did not get exact answer error; default to 0")
                original_preds.append(1)
        
        if task == "qqp":
            q1 = instance[col_ques1]
            q2 = instance[col_ques2]
            if instance[col_org_ques1] != '':
                original_question = instance[col_org_ques1]
                if isICR == True:
                    q1, q2 = create_multiinput_rewriting_prompt(examples_per_attack, q1, q2, model_id)
                original_prompt = f"Please identify whether Question1: \"{q2}\" has the same meaning as Question2: \"{original_question}\". The answer should only be exactly \"yes\" or \"no\". One word only, Nothing else."
            elif instance[col_org_ques2] != '':
                original_question = instance[col_org_ques2]
                if isICR == True:
                    q1, q2 = create_multiinput_rewriting_prompt(examples_per_attack, q1, q2, model_id)
                original_prompt = f"Please identify whether Question1: \"{q1}\" has the same meaning as Question2: \"{original_question}\". The answer should only be exactly \"yes\" or \"no\". One word only, Nothing else."

                

            prompt = f"Please identify whether Question1: \"{q1}\" has the same meaning as Question2: \"{q2}\". The answer should only be exactly \"yes\" or \"no\". One word only, Nothing else."
            response = model.invoke(prompt)
            original_response = model.invoke(original_prompt)

            response = response.lower().strip()
            original_response = original_response.lower().strip()
            # print(response, original_response, labels[i])
            if 'yes' in response:
                preds.append(1)
            elif 'no' in response:
                preds.append(0)
            else: 
                print("did not get exact answer error; default to 0")
                preds.append(0)

            if 'yes' in original_response:
                original_preds.append(1)
            elif 'no' in original_response:
                original_preds.append(0)
            else: 
                print("did not get exact answer error; default to 0")
                original_preds.append(0)
        
        if task == "mnli":
            premise = instance[col_premise]
            hypo = instance[col_hypo]
            if instance[col_org_premise] != '':
                original_premise = instance[col_org_premise]
                if isICR == True:
                    premise, hypo = create_multiinput_rewriting_prompt(examples_per_attack, premise, hypo, model_id)
                original_prompt = f"Please identify whether the premise: \"{original_premise}\" entails this hypothesis: \"{hypo}\". The answer should only be exactly \"yes\", \"maybe\", or \"no\". One word only, Nothing else."
            elif instance[col_org_hypo] != '':
                original_hypo = instance[col_org_hypo]
                if isICR == True:
                    premise, hypo = create_multiinput_rewriting_prompt(examples_per_attack, premise, hypo, model_id)
                original_prompt = f"Please identify whether the premise: \"{premise}\" entails this hypothesis: \"{original_hypo}\". The answer should only be exactly \"yes\", \"maybe\", or \"no\". One word only, Nothing else."

                

            prompt = f"Please identify whether the premise: \"{premise}\" entails this hypothesis: \"{hypo}\". The answer should only be exactly \"entailment\", \"neutral\", or \"contradiction\". One word only, Nothing else."
            response = model.invoke(prompt)
            original_response = model.invoke(original_prompt)

            response = response.lower().strip()
            original_response = original_response.lower().strip()
            # print(response, original_response, labels[i])
            if 'entailment' in response:
                preds.append(0)
            elif 'contradiction' in response:
                preds.append(2)
            elif 'neutral' in response:
                preds.append(1)
            else: 
                print("did not get exact answer error; default to 0")
                preds.append(0)

            if 'entailment' in response:
                preds.append(0)
            elif 'contradiction' in response:
                preds.append(2)
            elif 'neutral' in response:
                preds.append(1)
            else: 
                print("did not get exact answer error; default to 0")
                preds.append(0)
            
        # processed +=1
    
    if skipped > 0:
        print(f"Warning: {skipped} responses could not be parsed and were defaulted to negative")   

    return np.array(labels), np.array(preds)

def calculate_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict:
    """
    Calculate comprehensive metrics for model evaluation
    
    Args:
        labels: True labels
        preds: Model predictions
        
    Returns:
        Dictionary containing various metrics
    """
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics
            
def save_results(metrics: Dict, model_id: str, task: str, num_samples: int, 
                output_dir: str = "results"):
    """
    Save evaluation results to a CSV file with metadata
    
    Args:
        metrics: Dictionary of calculated metrics
        model_id: Model identifier
        task: Task name
        num_samples: Number of samples evaluated
        output_dir: Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results with metadata
    result_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_id': model_id,
        'task': task,
        'num_samples': num_samples,
        **metrics
    }
    
    # Convert to DataFrame
    df_new = pd.DataFrame([result_dict])
    
    # Define file path
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    
    # If file exists, append to it; otherwise create new file
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)
    
    print(f"Results saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Script to perform tasks with a specified model.")

    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="The ID of the model to be used."
    )

    parser.add_argument(
        "--task", 
        type=str, 
        required=True, 
        help="The task to be performed by the model."
    )

    args = parser.parse_args()
    model_id = args.model
    task = args.task

    print(model_id, task)

    ds = pd.read_csv(f"./benchmarks/advglueplusplus/advglueplusplus_{task}.csv")

    # tasks = ['sst2', 'qqp', 'mnli', 'qnli', 'rte']
    # datasets_map = {task:ds[task] for task in tasks}

    # # Display some sample data
    # print(datasets_map[tasks[0]].shape)

    task_to_keys = {
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
    }

    task_to_prompts = {
        "mnli": "unused",
        "mnli-mm": "unused",
        "qnli": "Does the sentence answer the question? The answer should be exactly \"yes\" or \"no\". Nothing else. ",
        "qqp": "unused",
        "rte": "unused",
        "sst2": "You must choose exactly one word from these two options: [\"positive\", \"negative\"]. Analyze this sentence and respond with only that one word, no punctuation or explanation: Sentence: ",
    }

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
        filtered = ds[ds["method"] == attack].head(1)

        # Filter the dataset to include only the required columns
        columns_to_include = ["index", "label"] + aux_col + ["method"]
        filtered_data = filtered[columns_to_include]

        final_data.append(filtered_data)
    
    final_data = pd.concat(final_data, ignore_index=True)
    labels, preds = evaluate(model_id,task,ds,task_to_keys,task_to_prompts,isICR=True)
    print(labels, preds)

    columns = ["index","pred","label","method"]
    res = pd.DataFrame(columns=columns)
    for i in range(len(final_data)):
        pred = preds[i]
        label = labels[i]
        method = final_data.iloc[i]["method"]  # Assuming `final_data` contains a 'method' field

        # Append the data to the DataFrame
        res = pd.concat(
            [
                res,
                pd.DataFrame([{"index": final_data.iloc[i]["index"], "pred": pred, "label": label, "method": method}])
            ],
            ignore_index=True
        )

    # Print or save the resulting DataFrame
    print(res)
    res.to_csv(f"llama27b_predictions_advplusplus_{task}.csv", index=False)

    
    # metrics = calculate_metrics(labels=labels, preds=preds)
    # asr = calculate_asr(original_preds,preds,labels)
    # metrics['asr'] = asr

    # save_results(metrics, model_id, task, num_samples)
    # print(processed)

if __name__ == "__main__":
    main()

