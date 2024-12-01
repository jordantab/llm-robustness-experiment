import logging
import argparse
from datetime import datetime
import json
import os
from ahp_evaluator import AHPEvaluator
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
        benchmark=args.benchmark
    )

    labels = []
    predictions = []
    checkpoint_interval = max(10, len(evaluator.dataset) // 10)
    
    # TODO: update the samples we are looping through
    for i, instance in enumerate(tqdm(evaluator.dataset, desc=f"Evaluating {args.benchmark}")):
        try:
            pred, label = evaluator.evaluate_sample(instance)
            if pred is not None and label is not None:
                predictions.append(pred)
                labels.append(label)
                
            if (i + 1) % checkpoint_interval == 0:
                metrics = evaluator.calculate_metrics(labels, predictions)
                logging.info(f"Progress: {i+1}/{len(evaluator.dataset)} samples. Current metrics: {metrics}")
                
        except Exception as e:
            logging.error(f"Error processing instance {i}: {str(e)}")
            continue

    # Calculate final metrics
    final_metrics = evaluator.calculate_metrics(labels, predictions)
    
    # Save results
    results = {
        'model_id': args.model_id,
        'benchmark': args.benchmark,
        'dataset_name': args.dataset_name,
        'num_samples': args.num_samples,
        'metrics': final_metrics,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    output_file = os.path.join(
        evaluator.checkpoint_dir,
        f"results_{args.model_id}_{args.benchmark}_{args.dataset_name}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Evaluation completed. Results saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run OOD AHP evaluation")
    parser.add_argument('--model_id', type=str, required=True, help="Model Identifier")
    parser.add_argument('--benchmark', type=str, choices=['flipkart', 'ddx'], required=True, help="Benchmark to use")

    args = parser.parse_args()
    results = run_evaluation(args)
    print(f"Final Metrics: {results['metrics']}")

if __name__ == "__main__":
    main()
