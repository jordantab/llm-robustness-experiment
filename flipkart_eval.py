"""
Baseline evaluation of model OOD Robustness using the Flipkart product review dataset.
"""
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from langchain_ollama.llms import OllamaLLM
import json
import csv
import os
from tqdm import tqdm


def import_dataset():
    dataset_path = "benchmarks/flipkart-sentiment.csv"
    return pd.read_csv(dataset_path)


def save_predictions(predictions, filename):
    with open(os.path.join(f"predictions/baseline/flipkart/{filename}"), 'w') as f:
        json.dump(predictions, f)


def load_predictions(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []


def ollama_invoke_model(review, model):
    """
    Use the provided model to generate a sentiment prediction for the given review.
    """
    # Define the prompt
    prompt = (
        "You are a world-class sentiment analyst. Respond ONLY in JSON format with a single field 'sentiment', "
        "which can be either 'positive', 'neutral', or 'negative'.\n"
        "Output format example: {'sentiment': 'negative'}\n\n"
        f"Analyze the sentiment of the following sentence without any explanation or additional text.\nSentence: {
            review}"
    )
    response = model.invoke(prompt)
    try:
        # Extract the JSON response
        response = response.replace("'", '"')
        response_json = json.loads(response)
        return response_json['sentiment']
    except (json.JSONDecodeError, KeyError):
        print(f"Error parsing response: {response}")
        return 'error'



def calculate_metrics(predictions, correct_labels):
    accuracy = accuracy_score(correct_labels, predictions)
    precision = precision_score(
        correct_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(
        correct_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(correct_labels, predictions,
                  average='weighted', zero_division=0)
    return accuracy, precision, recall, f1


def store_results(model_id, accuracy, precision, recall, f1):
    results_dir = "results/baseline"
    results_path = os.path.join(results_dir, "flipkart_results.csv")

    os.makedirs(results_dir, exist_ok=True)

    # Check if the CSV file already exists
    file_exists = os.path.isfile(results_path)

    with open(results_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if the file does not exist
        if not file_exists:
            writer.writerow(
                ["model_id", "accuracy", "precision", "recall", "f1"])

        # Write the results as a new row
        writer.writerow([model_id, accuracy, precision, recall, f1])


def run_experiment(df, model, model_id):
    filename = f"{model_id.replace('/', '_')}_predictions.json"

    correct_labels = df['Sentiment'].tolist()
    loaded_predictions = load_predictions(filename)
    predictions = loaded_predictions.copy()
    num_reviews = len(df['Summary'])

    with tqdm(total=num_reviews, desc="Processing") as pbar:
        for i, review in enumerate(df['Summary']):
            # Skip reviews if already seen
            if i < len(loaded_predictions):
                pbar.update(1)
                continue

            prediction = ollama_invoke_model(review, model)
            # prediction = bedrock_invoke_model(review)
            predictions.append(prediction)
            save_predictions(predictions, filename)
            pbar.update(1)

            # Calculate and print metrics every 10 reviews
            if (i + 1) % 10 == 0 or i == num_reviews - 1:
                accuracy, precision, recall, f1 = calculate_metrics(
                    predictions, correct_labels[:len(predictions)])

                tqdm.write(f'\nProgress: {i + 1}/{num_reviews} reviews processed.')
                tqdm.write(f'Accuracy: {accuracy:.2f}')
                tqdm.write(f'Precision: {precision:.2f}')
                tqdm.write(f'Recall: {recall:.2f}')
                tqdm.write(f'F1 Score: {f1:.2f}\n')

    # Final metrics after all predictions
    accuracy, precision, recall, f1 = calculate_metrics(
        predictions, correct_labels)

    tqdm.write(f'\nFinal Metrics:')
    tqdm.write(f'Accuracy: {accuracy:.2f}')
    tqdm.write(f'Precision: {precision:.2f}')
    tqdm.write(f'Recall: {recall:.2f}')
    tqdm.write(f'F1 Score: {f1:.2f}\n')
    return accuracy, precision, recall, f1


def main():
    '''
    Orchestrator
    '''
    df = import_dataset()
    filtered_df = df[df['Summary'].notna()]
    length_filtered_df = filtered_df[filtered_df['Summary'].str.len().between(
        150, 160)].head(300)
    model_id = "llama2:13b"
    model = OllamaLLM(model=model_id)
    accuracy, precision, recall, f1 = run_experiment(
        length_filtered_df, model, model_id)

    # Calculate existing results
    # actual_labels = length_filtered_df['Sentiment'].tolist()
    # predictions = load_predictions(f'{model_id}_predictions.json')
    # accuracy, precision, recall, f1 = calculate_metrics(
    #     predictions, actual_labels)

    # store the results
    store_results(model_id, accuracy, precision, recall, f1)


if __name__ == "__main__":
    main()
