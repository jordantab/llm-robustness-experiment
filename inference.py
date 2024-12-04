from tokens import TOKENS

from transformers import pipeline

import openai
import ollama

from huggingface_hub import login
from langchain_ollama.llms import OllamaLLM

import json

login("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")

class Inference(object):

    def __init__(self,
                 task,
                 service,
                 label_set,
                 model_set,
                 label_to_id,
                 model=None,    
                 device=0):  # service: hug, gpt, chat
        self.task = task
        self.service = service
        self.model = model
        self.label_set = label_set
        self.model_set = model_set
        self.label_to_id = label_to_id

        self.device = device if device else "cpu"
        
        if self.service == 'meta-huggingface':
            self.pipe = pipeline("text-generation", model=self.model, device=self.device)
    
        if self.service == 'openai':
            self.openai_client = openai.OpenAI(api_key=TOKENS['openai_api_key']) 

        if self.service == 'ollama-langchain':
            self.llm = OllamaLLM(model=self.model)
            
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
            response = ollama.chat(
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
            prediction = response['message']['content'].replace(".", '')

        if self.service == 'ollama-langchain':
            input = (
                f"{prompt}\n\n"
                f"Dialogue: {sentence}"
            )
            
            model_out = self.llm.invoke(f"{prompt} {sentence}")
            try:
                # Extract the JSON response
                response = model_out.replace("'", '"')
                response_json = json.loads(response)
                prediction = response_json['sentiment']
            except (json.JSONDecodeError, KeyError):
                print(f"Error parsing response: {response}")
                prediction = 'error'

        return prediction
    
if __name__=="__main__":
    ollama.pull("mistral")
    print(ollama.list())