from tokens import TOKENS

from transformers import pipeline

import openai
import ollama

from huggingface_hub import login
from langchain_ollama.llms import OllamaLLM

import json, os

login("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")

class Inference(object):

    def __init__(self,
                 task,
                 service,
                 label_set,
                 model=None, 
                 context_prompt=None,
                 rephrase_prompt=None,
                 device=0):  # service: hug, gpt, chat
        self.task = task
        self.service = service
        self.model = model
        self.label_set = label_set
        self.context_prompt = context_prompt
        self.rephrase_prompt = rephrase_prompt
        
        print("Context prompt: ", self.context_prompt)
        print("Rephrase prompt: ", self.rephrase_prompt)

        self.device = device if device else "cpu"
        
        if self.service == 'meta-huggingface':
            self.pipe = pipeline("text-generation", model=self.model, device=self.device)
    
        if self.service == 'openai':
            self.openai_client = openai.OpenAI(api_key=TOKENS['openai_api_key']) 

        if self.service == 'ollama-langchain':
            self.llm = OllamaLLM(model=self.model, num_ctx=8192)
            print("Context Length: ", self.llm.num_ctx)

        self.context = self.get_context(self.context_prompt)
        
    def get_context(self, context_prompt):
        # print("Generating context...")
        # print("context_prompt: ", context_prompt)
        # print("--------------------------------")
        file_path = './context.txt'
        
        context = ""
        
        if context_prompt and os.path.exists(file_path):
            with open(file=file_path, mode='r') as file:
                context = file.read()
                return context

        if context_prompt:
            for disease in self.label_set[:-1]:
                modified_context_prompt = context_prompt.replace("{disease}", disease)
                disease_description = self.llm.invoke(modified_context_prompt)
                disease_context = f"{disease}: {disease_description}\n"
                print(disease_context)
                context += disease_context

        # save context to a file
        file_path = './context.txt'
        with open(file=file_path, mode='w') as file:
            file.write(context)
        
        # print("--------------------------------")
        # print("Context: ", context)
        # print("--------------------------------")
        return context
        
    def predict(self, sentence, prompt):
        if self.service == 'meta-huggingface':
            messages=[
                {
                "role": "user",
                "content": f"{sentence}"
                },
                {
                "role": "system",
                "content": f"{prompt}"
                }
            ]
            prediction = self.pipe(messages)
        
        if self.service == 'openai':
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                    "role": "system",
                    "content": f"{prompt}"
                    },
                    {
                    "role": "user",
                    "content": f"{sentence}"
                    },
                    ]
                )
            prediction = response.choices[0].message.content
        
        if self.service == 'ollama':
            model_out = ollama.chat(
                model=self.model,
                messages=[
                    {
                    "role": "system",
                    "content": f"{prompt}"
                    },
                    {
                    "role": "user",
                    "content": f"{sentence}"
                    },
                    ]
                )
            model_out = model_out['message']['content'].replace(".", '')
            
            if "json" in self.task:
                if model_out[-1] != '}':
                    model_out = model_out + '}'
                # extract text from {}
                model_out_json_str = model_out[model_out.find('{'):model_out.rfind('}')+1]
                try:
                    # Extract the JSON response
                    response = model_out_json_str.replace("'", '"')
                    response_json = json.loads(response)
                    prediction = response_json['disease']
                except (json.JSONDecodeError, KeyError):
                    if ":" in model_out:
                        model_out = model_out[:-1]
                        prediction = model_out.split(":")[1]
                    else:
                        print(f"Error parsing response: {model_out}")
                        prediction = 'error'
            else:
                prediction = model_out


        if self.service == 'ollama-langchain':
            if self.rephrase_prompt:
                rephrase_prompt = self.rephrase_prompt.replace("{dialogue}", sentence)
                sentence = self.llm.invoke(rephrase_prompt)
                prompt = [prompt[0].replace("dialogue", "Patient Details")]
            input = ""
            
            if len(self.context) > 0:
                input += f"Here are short descriptions of diseases and their symptoms:\n{self.context}\n\n"
            
            input += f"{prompt[0]}\n\n"
            input += f"Patient Details: {sentence}"

            # print("Input: ")
            # print(input)

            model_out = self.llm.invoke(input)
            
            if "json" in self.task:
                if model_out[-1] != '}':
                    model_out = model_out + '}'
                # extract text from {}
                model_out_json_str = model_out[model_out.find('{'):model_out.rfind('}')+1]
                try:
                    # Extract the JSON response
                    response = model_out_json_str.replace("'", '"')
                    response_json = json.loads(response)
                    prediction = response_json['disease']
                except (json.JSONDecodeError, KeyError):
                    if ":" in model_out:
                        model_out = model_out[:-1]
                        prediction = model_out.split(":")[1]
                    else:
                        print(f"Error parsing response: {model_out}")
                        prediction = 'error'
            else:
                prediction = model_out

        return prediction
    
if __name__=="__main__":
    ollama.pull("mistral")
    print(ollama.list())