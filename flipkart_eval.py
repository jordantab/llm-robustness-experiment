"""
Baseline evaluation of model OOD Robustness using the Flipkart product review dataset.
"""
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import pipeline
from langchain_ollama.llms import OllamaLLM
import json
import os
import boto3
from tqdm import tqdm
from botocore.exceptions import ClientError


def import_dataset():
    dataset_path = "benchmarks/flipkart-sentiment.csv"
    return pd.read_csv(dataset_path)


def save_predictions(predictions, filename):
    with open(filename, 'w') as f:
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
        response_json = json.loads(response)
        return response_json['sentiment']
    except (json.JSONDecodeError, KeyError):
        print(f"Error parsing response: {response}")
        return 'neutral'


def bedrock_invoke_model(review):
    '''
    Model inference through AWS Bedrock
    '''
    prompt = f"You are a world class sentiment analyst that only outputs JSON. You reply in JSON format with the field 'sentiment'. The field can take one of three values: 'positive', 'neutral', 'negative'.\nExample Sentence: 'I hate this product'\nExample Answer: {{'sentiment': 'negative'}}. Now here is my question, what is the sentiment of the following sentence?\n Sentence: {
        review}"

    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    model_id = "mistral.mixtral-8x7b-instruct-v0:1"

    prompt = "Describe the purpose of a 'hello world' program in one line."

    native_request = {
        "anthropic_version": "bedrock-2024-02-29",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    model_response = json.loads(response["body"].read())

    response_text = model_response["content"][0]["text"]
    print(response_text)
    return response_text


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
                accuracy = accuracy_score(
                    correct_labels[:len(predictions)], predictions)
                precision = precision_score(
                    correct_labels[:len(predictions)], predictions, average='weighted')
                recall = recall_score(
                    correct_labels[:len(predictions)], predictions, average='weighted')
                f1 = f1_score(
                    correct_labels[:len(predictions)], predictions, average='weighted')

                tqdm.write(f'\nProgress: {
                           i + 1}/{num_reviews} reviews processed.')
                tqdm.write(f'Accuracy: {accuracy:.2f}')
                tqdm.write(f'Precision: {precision:.2f}')
                tqdm.write(f'Recall: {recall:.2f}')
                tqdm.write(f'F1 Score: {f1:.2f}\n')

    # Final metrics after all predictions
    accuracy = accuracy_score(correct_labels, predictions)
    precision = precision_score(
        correct_labels, predictions, average='weighted')
    recall = recall_score(correct_labels, predictions, average='weighted')
    f1 = f1_score(correct_labels, predictions, average='weighted')
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
        150, 160)]
    model_id = "mixtral"
    model = OllamaLLM(model=model_id)
    accuracy, precision, recall, f1 = run_experiment(
        length_filtered_df, model, model_id)

    # TODO: store the results


if __name__ == "__main__":
    main()
