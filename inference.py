from tokens import TOKENS

from transformers import pipeline

import openai
import ollama

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
        
        if self.service == 'mata-llama':
            self.pipe = pipeline(self.service, model=self.model, device=self.device)
    
        if self.service == 'openai':
            self.openai_client = openai.OpenAI(api_key=TOKENS['openai_api_key']) 


            
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

        return prediction
    
if __name__=="__main__":
    ollama.pull("mistral")
    print(ollama.list())