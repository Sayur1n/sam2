import ctypes
import os
import torch

print(f"Torch CUDA version: {torch.version.cuda}")
print("Searching loaded CUDA DLLs...\n")

dlls = [dll for dll in os.environ["PATH"].split(";") if "cuda" in dll.lower()]
for dll in dlls:
    print(dll)
