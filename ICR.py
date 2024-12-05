import numpy as np
from typing import List, Tuple, Dict, Union
from langchain_ollama.llms import OllamaLLM
from datasets import load_dataset
from huggingface_hub import login
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from datetime import datetime
import pandas as pd
import re
import json

# login("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")
def create_rewriting_prompt_plusplus(examples_per_attack, perturbed_sentence, model_id):
    """
    Create a prompt for in-context learning to rewrite perturbed sentences to cleaner ones.

    Parameters:
    - examples_per_attack: A dictionary where keys are attack types (e.g., 'semattack') and values are lists of examples.
        Each example is a tuple (perturbed_sentence, clean_sentence)
    - perturbed_sentence: The perturbed sentence that needs to be cleaned.

    Returns:
    - prompt: str
    """
    prompt = "Your task is to paraphrase the sentence while keeping semantic meaning. The sentence may be purturbed which means you will need generate their original cleaner form. If there are bad or unethical words, use semantically equivalent ethical synonym. \n\n"
    prompt += "Below are some examples:\n\n"

    for attack_type, examples in examples_per_attack.items():
        for example in examples:
            perturbed_ex, clean_ex = example
            prompt += f"[Attack Type: {attack_type}]\n"
            prompt += f"Perturbed Sentence: {perturbed_ex}\n"
            prompt += f"Cleaned Sentence: {clean_ex}\n\n"

    prompt += "Now, given the following sentence, please provide the paraphrased output. Give ONLY THE SENTENCE and nothing else.\n\n"
    prompt += f"Given Sentence: {perturbed_sentence}\n\n"
    prompt += (
    "Provide the response in valid JSON format, ensuring there are no syntax errors. "
    "Make sure to include both opening and closing braces. "
    "Only use one key: \"New Sentence\"."
    )

    model = OllamaLLM(model=model_id)
    res = model.invoke(prompt)
    # Ensure the response is valid JSON
    try:
        parsed_data = json.loads(res)
    except json.JSONDecodeError:
        # Fix common issues like missing closing brackets
        res_fixed = res.strip() + "}"
        try:
            parsed_data = json.loads(res_fixed)
        except:
            print(res_fixed)
            defalt = json.dumps({"New Sentence": perturbed_sentence})
            parsed_data = json.loads(defalt)
    # print(res)

    # Accessing specific inputs
    input_1 = parsed_data["New Sentence"]
    return input_1

def create_rewriting_prompt(examples_per_attack, perturbed_sentence, model_id):
    """
    Create a prompt for in-context learning to rewrite perturbed sentences to cleaner ones.

    Parameters:
    - examples_per_attack: A dictionary where keys are attack types (e.g., 'semattack') and values are lists of examples.
        Each example is a tuple (perturbed_sentence, clean_sentence)
    - perturbed_sentence: The perturbed sentence that needs to be cleaned.

    Returns:
    - prompt: str
    """
    prompt = "You are an expert linguist. Your task is to rewrite the instructions of the given task while keeping semantic meaning. The sentence may be purturbed like punctuation and grammar mistake which means you will need to generate a cleaner form. If there are bad or unethical words, use other semantically equivalent word. \n\n"
    prompt += "Below are some examples:\n\n"
    prompt += "Given Sentence: \"Analyze the sentiment of this text and respоnd with 'positive' or 'negative': the campy results make mel brooks\""
    prompt += "Pay attention to instruction and non-instruction portion of the sentence. You should do paraphrasing on the instruction itself and leave the non-instruction sentence as is. DO NOT DO THE TASK THE INSTRUCTION SAYS TO DO."

    for attack_type, examples in examples_per_attack.items():
        for example in examples:
            perturbed_ex, clean_ex = example
            prompt += f"[Attack Type: {attack_type}]\n"
            prompt += f"Perturbed Sentence: {perturbed_ex}\n"
            prompt += f"Cleaned Sentence: {clean_ex}\n\n"
    print(perturbed_sentence)
    prompt += "Now, given the following instruction, please rewrite the instruction. GIVE THE INSTRUCTION BACK. DO NOT JUST GIVE A SENTENCE. DO NOT perform the task specified in the Given Sentence section. JUST REWRITE THE GIVEN INSTRUCTION. DO NOT perform the task specified in the Given Sentence section. JUST REWRITE THE GIVEN INSTRUCTION. Give ONLY THE REWRITTEN INSTRUCTION and nothing else. ALSO Keep 'positive' and 'negative' EXACTLY AS THEY ARE.\n\n"
    prompt += f"Given Sentence: {perturbed_sentence}\n\n"
    prompt += (
    "Provide the response in valid JSON format, ensuring there are no syntax errors. "
    "Make sure to include both opening and closing braces. "
    "Only use one key: \"New Sentence\"."
    )

    model = OllamaLLM(model=model_id, temperature=0.0)
    res = model.invoke(prompt)
    # Ensure the response is valid JSON
    try:
        parsed_data = json.loads(res)
    except json.JSONDecodeError:
        # Fix common issues like missing closing brackets
        res_fixed = res.strip() + "}"
        parsed_data = json.loads(res_fixed)
    # print(res)

    # Accessing specific inputs
    input_1 = parsed_data["New Sentence"]
    return input_1

def create_multiinput_rewriting_prompt(examples_per_attack, input1, input2, model_id):
    """
    Create a prompt for in-context learning to rewrite perturbed sentences to cleaner ones.

    Parameters:
    - examples_per_attack: A dictionary where keys are attack types (e.g., 'semattack') and values are lists of examples.
        Each example is a tuple (perturbed_sentence, clean_sentence)
    - perturbed_sentence: The perturbed sentence that needs to be cleaned.

    Returns:
    - prompt: str
    """
    prompt = (
            "Your task is to paraphrase the two inputs below while keeping each of their semantic meaning in isolation. If the given input is a question, keep it a question."
            "Either or both of the inputs may be perturbed due to adversarial modifications. If they are purturbed, your goal is to "
            "restore them to their original, cleaner form.\n\n"
        )    
    prompt += "Below are some examples:\n\n"

    for attack_type, examples in examples_per_attack.items():
        for example in examples:
            perturbed_ex, clean_ex = example
            prompt += f"[Attack Type: {attack_type}]\n"
            prompt += f"Perturbed Sentence: {perturbed_ex}\n"
            prompt += f"Cleaned Sentence: {clean_ex}\n\n"

    prompt += "Now, given the following 2 inputs, please provide their paraphrased outputs. Give ONLY THE SENTENCE for each input and nothing else.\n\n"
    prompt += f"Given Input 1: {input1}\n\n"
    prompt += f"Given Input 2: {input2}\n\n"
    prompt += "Give answer in json with keys: \"New Input 1\" and \"New Input 2\". Remove trailing comma on the last key-value pair"

    model = OllamaLLM(model=model_id,  temperature=0.25, seed=42)
    res = model.invoke(prompt)
    try:
        parsed_data = json.loads(res)
    except json.JSONDecodeError:
        return (perturbed_sentence, perturbed_sentence2)

    # Accessing specific inputs
    input_1 = parsed_data["New Input 1"]
    input_2 = parsed_data["New Input 2"]
    return (input_1, input_2)

# Need to omit some rows to be used in prompt

# Define few-shot examples for each attack type
examples_per_attack = {
    'semattack': [
        ("allows us to hope that nolan not poised embark a major career as a commercial however inventive classmaker .", "allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker ."),
        ("defenden cau it to it good .", "holden caulfield did it better .")
    ],
    'textbugger': [
        ("allows us to hope that no lan is poised to embark a major career as a commercial yet infentive fmlimaker .", "allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker ."),
        ("heldon caulfield did it best .", "holden caulfield did it better .")
    ],
    'textfooler': [
        ("the mesmerizing operation of the leads keep the film grounded and keep the audience pore .", "the mesmerizing performances of the leads keep the film grounded and keep the audience riveted ."),
        ("holden caulfield did it well .", "holden caulfield did it better .")
    ],
    'sememepso': [
        ("unflinchingly bleak and desperate", "Why allopath typically approve herbation?"),
        ("... the movie is just a plain honest-to-goodness monster .", "... the movie is just a plain old monster .")
    ],
    'bertattack': [
        ("allows viewers to hope that nolan is ineligible to launch a major career as a successful yet inventive filmmaker.", "allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker"),
        ("holden caulfield took it better.", "holden caulfield did it better .")
    ],
}

def create_rewriting_prompt_OOD(examples, ood_sentence, model_id):
    """
    Create a prompt for in-context learning to rewrite perturbed sentences to cleaner ones.

    Parameters:
    - examples_per_attack: A dictionary where keys are attack types (e.g., 'semattack') and values are lists of examples.
        Each example is a tuple (perturbed_sentence, clean_sentence)
    - perturbed_sentence: The perturbed sentence that needs to be cleaned.

    Returns:
    - prompt: str
    """
    prompt = "Your task is to paraphrase the sentence while keeping its semantic meaning.\n\n"
    if examples is not None:
        prompt += "Below are some examples:\n\n"

        for example in examples:
            ood_ex, clean_ex = example
            prompt += f"Original Sentence: {ood_ex}\n"
            prompt += f"Cleaned Sentence: {clean_ex}\n\n"

    prompt += "Now, given the following sentence, please provide the paraphrased output. Give ONLY THE SENTENCE and nothing else.\n\n"
    prompt += f"Given Sentence: {ood_sentence}\n\n"
    prompt += (
    "Provide the response in valid JSON format, ensuring there are no syntax errors. "
    "Make sure to include both opening and closing braces. "
    "Only use one key: \"New Sentence\"."
    )

    model = OllamaLLM(model=model_id)
    res = model.invoke(prompt)
    # Ensure the response is valid JSON
    try:
        parsed_data = json.loads(res)
    except json.JSONDecodeError:
        # Fix common issues like missing closing brackets
        res_fixed = res.strip() + "}"
        parsed_data = json.loads(res_fixed)
    # print(res)

    # Accessing specific inputs
    input_1 = parsed_data["New Sentence"]
    return input_1
# The perturbed sentence that needs to be cleaned
perturbed_sentence = "less than one monthmoter automatically burnno replacement gurantee"
perturbed_sentence2 = "How do I block WhatsApp in my ship's wireless network?"

# # Generate the prompt
# prompt = create_multiinput_rewriting_prompt(examples_per_attack, perturbed_sentence, perturbed_sentence2)
# # prompt = create_rewriting_prompt(examples_per_attack, perturbed_sentence)

# print(prompt)

# model = OllamaLLM(model="llama2:7b",  temperature=0.0)
# res = model.invoke(prompt)
# print(res)

# # Parse the JSON
# parsed_data = json.loads(res)

# # Accessing specific inputs
# input_1 = parsed_data["New Input 1"]
# input_2 = parsed_data["New Input 2"]

# print("Input 1:", input_1)
# print("Input 2:", input_2)

# res1, res2 = create_multiinput_rewriting_prompt(examples_per_attack, perturbed_sentence, perturbed_sentence2)
# print(res1, res2)

# res3 = create_rewriting_prompt(examples_per_attack,perturbed_sentence)
# print(res3)

# res4 = create_rewriting_prompt_OOD(None, perturbed_sentence, "llama2:7b")
# print(res4)


