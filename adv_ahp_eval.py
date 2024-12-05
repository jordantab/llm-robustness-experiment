import logging
import argparse
from datetime import datetime
import json
import os
from ahp_evaluator2 import AHPEvaluator
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

def run_evaluation(args):
    evaluator = AHPEvaluator(
        model_id=args.model_id,
        robustness_type="adv",
        benchmark=args.benchmark,
        dataset_name=None,
        num_samples=args.num_samples
    )
    
    results_list = []
    labels = []
    predictions = []
    incomplete_count = 0
    formatting_count = 0
    checkpoint_interval = max(10, len(evaluator.dataset) // 10)
    total_samples = len(evaluator.dataset)
    
    for i, instance in enumerate(tqdm(evaluator.dataset, desc=f"Evaluating {args.benchmark}")):
        try:
            dataset_name = instance.get("dataset", "unknown")  # Extract dataset name dynamically
            evaluator.dataset_name = dataset_name
            
            pred, label, attack_method = evaluator.evaluate_sample(instance)
            if pred is not None and label is not None:
                # Handle predictions and labels
                if pred == "incomplete":
                    incomplete_count += 1
                elif pred == "formatting":
                    formatting_count += 1
                else:
                    predictions.append(pred)
                    labels.append(label)
                    
                results_list.append({
                    "original_prompt": instance.get("combined_prompt"),
                    "label": label,
                    "prediction": pred,
                    "dataset": dataset_name,
                    "attack_name": instance.get("attack_name")
                })
                
            # break
            if (i + 1) % checkpoint_interval == 0:
                metrics = evaluator.calculate_metrics(labels, predictions)
                logging.info(f"Progress: {i+1}/{len(evaluator.dataset)} samples. Current metrics: {metrics}")
        except Exception as e:
            logging.error(f"Error processing instance {i}: {str(e)}")
            continue

    # Calculate final metrics
    final_metrics = evaluator.calculate_metrics(labels, predictions)
    
    # Save detailed results to a CSV file
    results_df = pd.DataFrame(results_list)
    results_csv_path = os.path.join(
        evaluator.checkpoint_dir,
        f"detailed_results_{args.model_id}_{args.benchmark}.csv"
    )
    results_df.to_csv(results_csv_path, index=False)
    
    # Save results
    results = {
        'model_id': args.model_id,
        'robustness_type': args.robustness_type,
        'benchmark': args.benchmark,
        'metrics': final_metrics,
        'incomplete_AHP': incomplete_count,
        "formatting_mistakes": formatting_count,
        'num_samples': args.num_samples,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    output_file = os.path.join(
        evaluator.checkpoint_dir,
        f"results_{args.model_id}_{args.benchmark}_{args.dataset_name}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Evaluation completed. Results saved to {output_file}")
    logging.info(f"Detailed results saved to {results_csv_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Run AHP evaluation')
    parser.add_argument('--model_id', type=str, required=True, help='Model identifier')
    parser.add_argument('--benchmark', type=str, choices=['promptbench', 'advglue++'], 
                       required=True, help='Benchmark to use')
    # parser.add_argument('--dataset_name', type=str, default='sst2', 
                    #    help='Dataset name (default: sst2)')
    parser.add_argument('--robustness_type', type=str, required=True, help="Adv or OOD", default="adv")
    parser.add_argument('--num_samples', type=int, default=50, 
                       help='Number of samples to evaluate')
    
    args = parser.parse_args()
    results = run_evaluation(args)
    print(f"Final metrics: {results['metrics']}")

if __name__ == "__main__":
    main()