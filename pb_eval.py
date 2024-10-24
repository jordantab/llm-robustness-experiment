import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import time
import json
from typing import Dict, List, Union
from langchain_ollama.llms import OllamaLLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from promptbench.prompt_attack import Attack
from tqdm import tqdm

# Attack using PromptRobust, implementation in promptbench
# paper: https://arxiv.org/abs/2306.04528

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

class PromptRobustnessEvaluator:
    def __init__(self, model_id: str, dataset_name: str = "sst2", num_samples: int = None, checkpoint_dir: str = "checkpoints"):
        """
        Initialize evaluator with enhanced error handling and checkpoint support
        """
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            self.model = OllamaLLM(model=model_id)
            self.dataset = load_dataset("glue", dataset_name)["validation"]
            if num_samples:
                self.dataset = self.dataset.select(range(num_samples))
        except Exception as e:
            logging.error(f"Failed to initialize model or dataset: {str(e)}")
            raise

        self.task_prompts = {
            "sst2": "Analyze the sentiment of this text and respond with 'positive' or 'negative': {content}",
            "qnli": "Does this sentence answer the question? Respond with 'yes' or 'no': Question: {question} Sentence: {sentence}",
            "qqp": "Are these questions asking the same thing? Respond with 'yes' or 'no': Question 1: {question1} Question 2: {question2}",
            "mnli": "Does the premise entail the hypothesis? Respond with 'entailment', 'neutral', or 'contradiction': Premise: {premise} Hypothesis: {hypothesis}"
        }
        
        self.label_maps = {
            "sst2": {0: "negative", 1: "positive"},
            "qnli": {0: "no", 1: "yes"},
            "qqp": {0: "no", 1: "yes"},
            "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"}
        }

    def save_checkpoint(self, results: Dict, step: int, phase: str):
        """Save evaluation checkpoint"""
        try:
            checkpoint_file = os.path.join(
                self.checkpoint_dir, 
                f"checkpoint_{self.model_id}_{self.dataset_name}_{phase}_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(checkpoint_file, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {str(e)}")

    def process_model_response(self, response: str, max_retries: int = 3) -> int:
        """Parse model response with retry logic"""
        for attempt in range(max_retries):
            try:
                response = response.lower().strip()
                label_map = {v.lower(): k for k, v in self.label_maps[self.dataset_name].items()}
                
                for key in label_map:
                    if key in response:
                        return label_map[key]
                return None
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to process response after {max_retries} attempts: {str(e)}")
                    return None
                time.sleep(1)  # Wait before retry

    def evaluate_clean(self) -> tuple:
        """Evaluate model on clean samples with checkpointing"""
        labels = []
        predictions = []
        results = []
        checkpoint_interval = max(10, len(self.dataset) // 10)  # Save every 10% of progress
        
        for i, instance in enumerate(tqdm(self.dataset, desc="Clean evaluation")):
            try:
                # Format prompt based on dataset
                if self.dataset_name == "sst2":
                    prompt = self.task_prompts[self.dataset_name].format(content=instance["sentence"])
                elif self.dataset_name == "qnli":
                    prompt = self.task_prompts[self.dataset_name].format(
                        question=instance["question"],
                        sentence=instance["sentence"]
                    )
                # Add other dataset formats as needed
                
                response = self.model.invoke(prompt)
                pred = self.process_model_response(response)
                
                if pred is not None:
                    predictions.append(pred)
                    labels.append(instance["label"])
                    results.append({
                        "instance_id": i,
                        "prompt": prompt,
                        "response": response,
                        "prediction": pred,
                        "label": instance["label"]
                    })
                
                # Save checkpoint periodically
                if (i + 1) % checkpoint_interval == 0:
                    metrics = self.calculate_metrics(np.array(labels), np.array(predictions))
                    checkpoint_data = {
                        "step": i + 1,
                        "results": results,
                        "current_metrics": metrics
                    }
                    self.save_checkpoint(checkpoint_data, i + 1, "clean")
                    
            except Exception as e:
                logging.error(f"Error processing instance {i}: {str(e)}")
                continue
                
        return np.array(labels), np.array(predictions)

    def evaluate_adversarial(self) -> Dict:
        """Evaluate model against adversarial prompts"""
        attack_results = {}
        
        def eval_func(prompt, dataset, model):
            preds = []
            labels = []
            results = []
            
            for i, d in enumerate(dataset):
                try:
                    input_text = prompt.format(content=d["sentence"])
                    response = model.invoke(input_text)
                    pred = self.process_model_response(response)
                    if pred is not None:
                        preds.append(pred)
                        labels.append(d["label"])
                        results.append({
                            "instance_id": i,
                            "input": input_text,
                            "response": response,
                            "prediction": pred,
                            "label": d["label"]
                        })
                except Exception as e:
                    logging.error(f"Error in eval_func for instance {i}: {str(e)}")
                    continue
                    
            return accuracy_score(labels, preds) if preds else 0.0

        for attack_name in tqdm(Attack.attack_list(), desc="Running attacks"):
            try:
                attack = Attack(
                    self.model,
                    attack_name,
                    self.dataset,
                    self.task_prompts[self.dataset_name],
                    eval_func,
                    unmodifiable_words=list(self.label_maps[self.dataset_name].values()),
                    verbose=True
                )
                result = attack.attack()
                attack_results[attack_name] = result
                
                # Save checkpoint after each attack
                self.save_checkpoint(
                    {"attack_name": attack_name, "result": result},
                    len(attack_results),
                    "adversarial"
                )
                
            except Exception as e:
                logging.error(f"Error in {attack_name}: {str(e)}")
                attack_results[attack_name] = {"error": str(e)}

        return attack_results

    def calculate_metrics(self, labels: np.ndarray, preds: np.ndarray) -> Dict:
        """Calculate evaluation metrics with error handling"""
        try:
            metrics = {
                'accuracy': accuracy_score(labels, preds),
                'precision': None,
                'recall': None,
                'f1': None
            }
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average='weighted', zero_division=0
            )
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            return metrics
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return {
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1': None,
                'error': str(e)
            }

def main():
    # Configuration
    models = ["mixtral", "llama2"]
    datasets = ["sst2"]
    num_samples = 300
    
    for model_id in models:
        for dataset_name in datasets:
            logging.info(f"Starting evaluation of {model_id} on {dataset_name}")
            
            try:
                evaluator = PromptRobustnessEvaluator(
                    model_id=model_id,
                    dataset_name=dataset_name,
                    num_samples=num_samples
                )
                
                # Clean evaluation
                clean_labels, clean_preds = evaluator.evaluate_clean()
                clean_metrics = evaluator.calculate_metrics(clean_labels, clean_preds)
                
                # Adversarial evaluation
                adv_results = evaluator.evaluate_adversarial()
                
                # Combine and save final results
                results = {
                    'clean_metrics': clean_metrics,
                    'adversarial_results': adv_results
                }
                evaluator.save_checkpoint(results, -1, "final")  # -1 indicates final results
                
            except Exception as e:
                logging.error(f"Fatal error evaluating {model_id} on {dataset_name}: {str(e)}")
                continue

if __name__ == "__main__":
    main()