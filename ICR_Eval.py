from ICR import * 
from huggingface_hub import login 
from langchain_ollama.llms import OllamaLLM
import pandas as pd
import json
import argparse

def load_dataset_for_benchmark(benchmark): 
    if benchmark == "promptbench":
        path = './benchmarks/promptattack.csv'
        data = pd.read_csv(path)

        # Extract the `combined_prompt` and `label` columns
        extracted_data = data[["combined_prompt", "label", "idx"]]
        return extracted_data
    elif benchmark == "flipkart":
        path = './benchmarks/flipkart-sentiment.csv'
        df = pd.read_csv(path)
        filtered_df = df[df['Summary'].notna()]
        length_filtered_df = filtered_df[filtered_df['Summary'].str.len().between(
            150, 160)]

        # Extract the `Summary` and `Sentiment` columns
        extracted_data = length_filtered_df[["Summary", "Sentiment"]]
        return extracted_data
    else: 
        print("Not a valid benchmark")

def evaluate_with_ICR(benchmark, model_id):
    model = OllamaLLM(model=model_id)
    
    if benchmark == "promptbench":
        data = load_dataset_for_benchmark(benchmark)
        columns = ["idx", "prompt", "pred", "true_label"]
        res = pd.DataFrame(columns=columns)
        for instance in data: 
            combined_prompt = instance["combined_prompt"] 
            combined_prompt_ICR = create_rewriting_prompt(examples_per_attack,combined_prompt,model_id)

            response = model.invoke(combined_prompt_ICR)
            response = response.strip().lower()
            if "positive" in response:
                pred = 1
            elif "negative" in response: 
                pred = 0
            else: 
                print("Undetermined: Default to negative")
                pred = 0
            new_row = {"idx": instance["idx"], "prompt": combined_prompt_ICR, "pred": pred, "true_label": instance["label"]}
            res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)
        
        return res
    
    elif benchmark == "flipkart":
        data = load_dataset_for_benchmark(benchmark)
        columns = ["summary_icr", "pred", "true_label"]
        res = pd.DataFrame(columns=columns)
        for i in range(len(data)): 
            instance = data.iloc[i]
            summary = instance["Summary"] 
            summary_ICR = create_rewriting_prompt_OOD(None, summary, model_id)
            # print(summary_ICR)
            prompt = (
                "You are a world-class sentiment analyst. Respond ONLY in JSON format with a single field 'sentiment', "
                "which can be either 'positive', 'neutral', or 'negative'.\n"
                "Output format example: {'sentiment': 'negative'}\n\n"
                f"Analyze the sentiment of the following sentence without any explanation or additional text.\nSentence: {
                    summary_ICR}"
            )
            response = model.invoke(prompt)

            response = response.replace("'", '"')
            response_json = json.loads(response)
            sentiment = response_json['sentiment']
            sentiment = sentiment.strip().lower()
            new_row = {"summary_icr": summary_ICR, "pred": sentiment, "true_label": instance["Sentiment"]}
            res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)
            if i % 20 == 0:
                print(f"Progress: {i}/{len(data)}")
        return res

    else: 
        print("Not a valid benchmark")

def calculate_metrics(label, pred):
    pass

def main(): 
    parser = argparse.ArgumentParser(description="Script to perform tasks with a specified model.")

    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="The ID of the model to be used."
    )

    parser.add_argument(
        "--bm", 
        type=str, 
        required=True, 
        help="The benchmark to be performed by the model."
    )

    args = parser.parse_args()
    model_id = args.model
    benchmark = args.bm

    print(model_id, benchmark)

    # res is a Dataframe of results
    res = evaluate_with_ICR(benchmark, model_id)

    # store dataframe 
    path = f"./predictions/{model_id}_predictions_{benchmark}_ICR.csv"
    res.to_csv(path, index=False)

    # now calculate metrics 







if __name__ == "__main__":
    df = load_dataset_for_benchmark("flipkart")
    print(len(df))
    main()