from ICR import * 
from langchain_ollama.llms import OllamaLLM
import pandas as pd
import json
import argparse

def load_dataset_for_benchmark(benchmark): 
    if benchmark == "promptbench":
        path = './benchmarks/promptattack_reduced.csv'
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
        length_filtered_df = length_filtered_df[:300]
        # Extract the `Summary` and `Sentiment` columns
        extracted_data = length_filtered_df[["Summary", "Sentiment"]]
        return extracted_data
    else: 
        print("Not a valid benchmark")

def evaluate_with_ICR(benchmark, model_id):
    model = OllamaLLM(model=model_id, temperature=0.0)
    
    if benchmark == "promptbench":
        data = load_dataset_for_benchmark(benchmark)
        columns = ["idx", "prompt", "pred", "true_label"]
        res = pd.DataFrame(columns=columns)
        # Define the output file path
        output_file = f"./predictions/YOW_predictions_{benchmark}_ICR.csv"
        
        # Write headers only if the file doesn't exist
        if not os.path.exists(output_file):
            res.to_csv(output_file, index=False, mode='w')  # Write headers

        for i in range(len(data)): 
            instance = data.iloc[i]
            combined_prompt = instance["combined_prompt"] 
            combined_prompt_ICR = create_rewriting_prompt(examples_per_attack,combined_prompt,model_id)
            print(combined_prompt_ICR)
            try:
                response = model.invoke(combined_prompt_ICR + ". Respond in ONE WORD ONLY AND NOTHING ELSE with either \"positive\" or \"negative\" in json format with only one key \"sentiment\"")
                response = response.strip().lower()
            except: 
                response = "{\"sentiment\": \"Error\"}"

            try:
                parsed_data = json.loads(response)
            except json.JSONDecodeError:
                # Fix common issues like missing closing brackets
                try:
                    print(response)
                    res_fixed = response + "}"
                    parsed_data = json.loads(res_fixed)
                except json.JSONDecodeError:
                    parsed_data = json.loads("{\"sentiment\": \"Error\"}")
            print(parsed_data["sentiment"], "\n")
            sentiment = parsed_data["sentiment"]
            # if "positive" in response:
            #     pred = 1
            # elif "negative" in response: 
            #     pred = 0
            # else: 
            #     print("Undetermined: Default to negative")
            #     pred = 0
            new_row = {"idx": instance["idx"], "prompt": combined_prompt, "pred": sentiment, "true_label": instance["label"]}
            res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)

            if (i + 1) % 10 == 0:
                print(f"Saving intermediate results to {output_file}")
                res.to_csv(output_file, index=False, mode='a', header=False)  # Append without headers
                res = pd.DataFrame(columns=columns)  # Clear the in-memory DataFrame
        
        if not res.empty:
            print(f"Saving final results to {output_file}")
            res.to_csv(output_file, index=False, mode='a', header=False)
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
                f"Analyze the sentiment of the following sentence without any explanation or additional text.\nSentence: {summary_ICR}"
            )
            response = model.invoke(prompt)

            response = response.replace("'", '"')
            print(response)
            response_json = json.loads(response)
            sentiment = response_json['sentiment']
            sentiment = sentiment.strip().lower()
            print(sentiment)
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
    # df = load_dataset_for_benchmark("flipkart")
    # print(len(df))
    main()