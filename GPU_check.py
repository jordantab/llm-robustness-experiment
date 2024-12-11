import torch
print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda)  # Check the CUDA version PyTorch is using
print(torch.cuda.is_available())  # Confirm if PyTorch detects CUDA
