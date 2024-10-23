"""
Baseline evaluation of model OOD Robustnes using the Flipkart product review dataset
"""
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import pipeline

from dataloader import DataDDXPlus
from inference import Inference

from tqdm import tqdm
# TODO: Modularity



def run_ddxplus_experiment(dataloader: DataDDXPlus, inference: Inference):
    data_len = len(dataloader.get_data_by_task(dataloader.task))

    prompt = dataloader.get_prompt()

    predictions = []
    ground_truth = []
    for idx in tqdm(range(data_len)):
        information, correct_label = dataloader.get_content_by_idx(idx)
        prediction = inference.predict(information, prompt)
        predictions.append(prediction.lower())
        ground_truth.append(correct_label.lower())

        # if prediction.lower() != correct_label.lower():
        #     print(f"Prediction - {prediction} | Ground Truth - {correct_label}")
            
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(
        ground_truth, predictions, average='weighted')
    recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
    f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
    print(f'Accuracy Score: {accuracy:.2f}')
    print(f'Precision Score: {precision:.2f}')
    print(f'Recall Score: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1
    }
    return metrics

def main():
    '''
    Orchestrator
    '''
    dataloader = DataDDXPlus("benchmarks/ddxplus-hao.csv", task='ddxplus')
    # print("Prompt - ", dataloader.get_prompt(), "\n")
    print("Label Set - ", dataloader.get_label(), "\n")
    # print("Example data (idx 0) - ", dataloader.get_content_by_idx(0), "\n")

    metric_log = [
        ["model name", "model service", "accuracy", "precision", "recall", "f1-score"] 
    ]

    # Models - Service
    MODEL_LIST = \
    [
        # {
        #     "model" : "meta-llama/Llama-2-7b-chat-hf",
        #     "service": "meta-huggingface"
        # },
        # {
        #     "model" : "gpt-3.5-turbo",
        #     "service": "openai"
        # },
        {
            "model" : "llama3.2",
            "service": "ollama"
        },
        {
            "model" : "mistral",
            "service": "ollama"
        },
        {
            "model" : "mixtral",
            "service": "ollama"
        },
        {
            "model" : "llama2",
            "service": "ollama"
        },
        {
            "model" : "phi",
            "service": "ollama"
        },
    ]

    for llm_model in MODEL_LIST:
        print(f"Running experiment for {llm_model['model']} using {llm_model['service']}")
        inference = Inference(
            task='ddxplus',
            service=llm_model["service"],
            model=llm_model["model"],
            label_set=dataloader.get_label(),
            model_set=None,
            label_to_id=None,
            device=0
        )
        metrics = run_ddxplus_experiment(dataloader, inference)
        metric_log.append([llm_model["model"], llm_model["service"], metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1-score"]])
    
    # Save to CSV
    df = pd.DataFrame(metric_log)
    df.to_csv("ddxplus_eval.csv", index=False)


if __name__ == "__main__":
    main()
