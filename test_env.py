import torch
import cv2
import sys

print("--- Environment Check ---")
print("Python Version:", sys.version)
print("Torch Version:", torch.__version__)
print("OpenCV Version:", cv2.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("Torch OK ✔ (GPU)")
else:
    print("Torch OK ✔ (CPU)")

print("OpenCV OK ✔")
print("--------------------------")
