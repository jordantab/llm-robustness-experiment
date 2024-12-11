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
    def __init__(self, model_id: str, dataset_name: str = "sst2", num_samples: int = None, checkpoint_dir: str = "checkpoints_baseline"):
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
        """Evaluate model against adversarial prompts with both periodic and attack-level checkpointing"""
        attack_results = {}
        
        # Define reduced attack configurations to limit variations
        reduced_attack_config = {
            "textbugger": {
                "max_candidates": 3,
                "min_sentence_cos_sim": 0.8,
            },
            "deepwordbug": {
                "levenshtein_edit_distance" : 20,
            },
            "bertattack": {
                "max_candidates": 10,
                "max_word_perturbed_percent": 0.5,
                "min_sentence_cos_sim": 0.8,
            },
            "checklist": {
                "max_candidates": 2,
            }
        }
        
        # reduced_attack_config = {
        #     "textbugger": {
        #         "max_candidates": 3,  # Increase candidates
        #         "min_sentence_cos_sim": 0.6,  # Slightly lower similarity constraint
        #     },
        #     "deepwordbug": {
        #         "levenshtein_edit_distance": 50,  # Increase allowed edit distance
        #     },
        #     "bertattack": {
        #         "max_candidates": 3,  # More candidate replacements
        #         "max_word_perturbed_percent": 0.8, # Allow more words to be changed
        #         "min_sentence_cos_sim": 0.5,  # Much lower semantic similarity requirement
        #     },
        #     "checklist": {
        #         "max_candidates": 5,  # More candidates for sentence-level transformations
        #     }
        # }

        
        def eval_func(prompt, dataset, model):
            preds = []
            labels = []
            results = []
            checkpoint_interval = max(10, len(dataset) // 10)

            # Create/load checkpoint file for current attack
            attack_checkpoint_file = os.path.join(
                self.checkpoint_dir,
                f"adv_attack_{self.model_id}_{self.dataset_name}_{attack_name}_progress.json"
            )
            
            # Load existing results if any
            try:
                with open(attack_checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    all_prompt_results = checkpoint_data.get('prompt_results', {})
            except (FileNotFoundError, json.JSONDecodeError):
                all_prompt_results = {}

            # Start evaluating current prompt
            for i, d in enumerate(tqdm(dataset, desc="Evaluating adversarial prompt")):
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
                    
                    # Update checkpoint every 10% of samples
                    if (i + 1) % checkpoint_interval == 0:
                        current_acc = accuracy_score(labels, preds) if preds else 0.0
                        logging.info(f"Progress: {i+1}/{len(dataset)} samples. Current accuracy: {current_acc:.3f}")
                        
                        # Update results for current prompt
                        all_prompt_results[prompt] = {
                            "completed_samples": i + 1,
                            "current_accuracy": current_acc,
                            "results": results,
                            "last_update": datetime.now().strftime('%Y%m%d_%H%M%S')
                        }
                        
                        # Save updated checkpoint
                        checkpoint_data = {
                            "model_id": self.model_id,
                            "dataset": self.dataset_name,
                            "attack_name": attack_name,
                            "total_samples": len(dataset),
                            "prompt_results": all_prompt_results,
                            "last_update": datetime.now().strftime('%Y%m%d_%H%M%S')
                        }
                        
                        with open(attack_checkpoint_file, 'w') as f:
                            json.dump(checkpoint_data, f, indent=2)
                        logging.info(f"Updated checkpoint for {attack_name} with new prompt results")
                        
                except Exception as e:
                    logging.error(f"Error in eval_func for instance {i}: {str(e)}")
                    continue
            
            # Save final results for this prompt
            final_acc = accuracy_score(labels, preds) if preds else 0.0
            all_prompt_results[prompt] = {
                "completed_samples": len(dataset),
                "final_accuracy": final_acc,
                "results": results,
                "last_update": datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
            # Save final checkpoint for this prompt
            checkpoint_data = {
                "model_id": self.model_id,
                "dataset": self.dataset_name,
                "attack_name": attack_name,
                "total_samples": len(dataset),
                "prompt_results": all_prompt_results,
                "last_update": datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
            with open(attack_checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            return final_acc

        # Use only selected attacks
        selected_attacks = [
            # 'textbugger',    # Character-level
            # 'deepwordbug',   # Character-level  
            # 'textfooler',    # Word-level
            # 'bertattack',    # Word-level
            # 'checklist',     # Sentence-level
            # 'stresstest',    # Sentence-level
            'semantic'       # Semantic-level
        ]
        
        # Protected words that must not be modified
        protected_words = {
            # Label keywords with variations needed for attack constraints
            "positive", "negative", "positive\'", "negative\'", 
            # Format placeholders
            "content", "question", "sentence"
        }

        # Track overall progress
        total_attacks = len(selected_attacks)
        
        for attack_idx, attack_name in enumerate(selected_attacks, 1):
            try:
                logging.info(f"\nStarting attack {attack_idx}/{total_attacks}: {attack_name}")
                
                attack = Attack(
                    self.model,
                    attack_name,
                    self.dataset,
                    self.task_prompts[self.dataset_name],
                    eval_func,
                    unmodifiable_words=list(protected_words),
                    verbose=True  # Keep verbose for monitoring progress
                )
                
                result = attack.attack()
                attack_results[attack_name] = result
                
                # Save comprehensive checkpoint after each attack
                checkpoint_data = {
                    "model_id": self.model_id,
                    "dataset": self.dataset_name,
                    "attack_name": attack_name,
                    "attack_index": attack_idx,
                    "total_attacks": total_attacks,
                    "result": result,
                    "protected_words": list(protected_words),
                    "attack_config": reduced_attack_config.get(attack_name, {}),
                    "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
                }
                
                # Save both periodic checkpoint and attack result
                checkpoint_file = os.path.join(
                    self.checkpoint_dir, 
                    f"adv_attack_{self.model_id}_{self.dataset_name}_{attack_name}.json"
                )
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                logging.info(f"Completed and saved results for {attack_name} attack ({attack_idx}/{total_attacks})")
                
            except Exception as e:
                logging.error(f"Error in {attack_name}: {str(e)}")
                attack_results[attack_name] = {"error": str(e)}

        # Save final combined results
        final_results = {
            "model_id": self.model_id,
            "dataset": self.dataset_name,
            "all_attack_results": attack_results,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        final_file = os.path.join(
            self.checkpoint_dir,
            f"adv_attacks_final_{self.model_id}_{self.dataset_name}.json"
        )
        
        with open(final_file, 'w') as f:
            json.dump(final_results, f, indent=2)

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
    models = ["llama2:13b"]
    datasets = ["sst2"]
    num_samples = 50
    
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