"""
Baseline evaluation of model OOD Robustnes using the Flipkart product review dataset
"""
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import pipeline

# TODO: Modularity


def import_dataset():
    dataset_path = "benchmarks/flipkart-sentiment.csv"

    df = pd.read_csv(dataset_path)
    return df


# TODO: invoke model
def invoke_model(review):
    messages = [
        {"role": "user", "content": f"Is the following sentence positive, neutral, or negative?  Answer me with 'positive', 'neutral', or 'negative', just one word. {review}"},
    ]
    return 'positive'
    pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
    pipe(messages)
    return pipe


def run_experiment(df):

    filtered_df = df[df['Summary'].notna()]
    # print(len(filtered_df))
    # print(correct_labels)
    correct_labels = filtered_df['Sentiment'].tolist()

    long_reviews = []
    predictions = []
    for review in filtered_df['Summary']:
        # if len(review) > 150 and len(review) < 160:  # 1000 reviews
        # if len(review) > 150:  # 12857 reviews
        # TODO: invoke model to make prediction on each input in the dataset
        prediction = invoke_model(review)
        predictions.append(prediction)

        long_reviews.append(review)

    # print(len(long_reviews))
    # calculate accuracy, precision, recall, F1-score
    accuracy = accuracy_score(correct_labels, predictions)
    precision = precision_score(
        correct_labels, predictions, average='weighted')
    recall = recall_score(correct_labels, predictions, average='weighted')
    f1 = f1_score(correct_labels, predictions, average='weighted')
    print(f'Accuracy Score: {accuracy:.2f}')
    print(f'Precision Score: {precision:.2f}')
    print(f'Recall Score: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')


# TODO: store results


def main():
    '''
    Orchestrator
    '''
    df = import_dataset()
    # print(df.head())
    run_experiment(df)


if __name__ == "__main__":
    main()
