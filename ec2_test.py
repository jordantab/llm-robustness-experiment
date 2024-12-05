import torch
from transformers import pipeline


def cuda_test():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")


def transformer_test():
    print(2)
    try:
        generator = pipeline(
            "text-generation", 
            model="meta-llama/Llama-2-7b-hf", 
            device_map="auto"
        )
        print("Transformers library is working correctly!")
        
        # Generate text
        output = generator("The Transformers library is", max_length=30, num_return_sequences=1)
        print("Generated text:")
        print(output[0]["generated_text"])
    except Exception as e:
        print("There was an error testing the transformers library:")
        print(e)


if __name__ == "__main__":
    print(1)
    # cuda_test()
    transformer_test()