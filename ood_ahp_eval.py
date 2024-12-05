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
        robustness_type=args.robustness_type,
        benchmark=args.benchmark
    )

    labels = []
    predictions = []
    incomplete_count = 0
    formatting_count = 0
    checkpoint_interval = min(3, len(evaluator.dataset) // 10)
    total_samples = len(evaluator.dataset)
    # print(evaluator.dataset)
    # print(evaluator.dataset['Diagnosis'].value_counts())
    # return

    # Create the tqdm progress bar
    with tqdm(total=total_samples, desc=f"Evaluating {args.benchmark}", unit="sample") as pbar:
        for i, row in enumerate(evaluator.dataset.iterrows()):
            try:
                # print(i)
                if args.benchmark == "flipkart":
                    instance = {'Summary': row[1]["Summary"], 'Sentiment': row[1]["Sentiment"]}
                else:
                    instance = {'Information': row[1]["Information"], "Diagnosis": row[1]["Diagnosis"]}
                pred, label = evaluator.evaluate_sample(instance)
                # print("##"*100)
                # break
                print(pred, label)
                # break
                # Failed during AHP process
                if pred == "incomplete":
                    incomplete_count += 1
                
                # Final prediction failed formatting
                elif pred == "formatting":
                    formatting_count += 1

                # Successfully made a prediction
                else:
                    predictions.append(pred)
                    labels.append(label)

                # Update progress bar
                pbar.update(1)

                # Checkpoint logging
                if (i + 1) % checkpoint_interval == 0:
                    metrics = evaluator.calculate_metrics(labels, predictions)
                    logging.info(f"Progress: {i+1}/{total_samples} samples. Incomplete AHP count: {incomplete_count}. Formatting error count: {formatting_count}. Current metrics: {metrics}.")
            except Exception as e:
                logging.error(f"Error processing instance {i}: {str(e)}")
                continue

    # Calculate final metrics
    final_metrics = evaluator.calculate_metrics(labels, predictions)

    # Save results
    results = {
        'model_id': args.model_id,
        'robustness_type': args.robustness_type,
        'benchmark': args.benchmark,
        'metrics': final_metrics,
        'incomplete_AHP': incomplete_count,
        "formatting_mistakes": formatting_count,
        'num_samples': total_samples, 
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    output_file = os.path.join(
        evaluator.checkpoint_dir,
        f"{args.robustness_type}_{args.model_id}_{args.benchmark}.json"
    )

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Evaluation completed. Results saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run OOD AHP evaluation")
    parser.add_argument('--model_id', type=str, required=True, help="Model Identifier")
    parser.add_argument('--robustness_type', type=str, required=True, help="Adv or OOD")
    parser.add_argument('--benchmark', type=str, choices=['promptbench','flipkart', 'ddx'], required=True, help="Benchmark to use")

    args = parser.parse_args()
    results = run_evaluation(args)
    print(f"Final Metrics: {results['metrics']}")

if __name__ == "__main__":
    main()