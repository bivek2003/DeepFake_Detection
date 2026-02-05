import torch
import sys

print(f"Loading checkpoint from {sys.argv[1]}")
try:
    ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
except Exception as e:
    print(f"Error loading: {e}")
    sys.exit(1)

state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

print("Keys and shapes:")
for k, v in state_dict.items():
    if "classifier" in k:
        print(f"{k}: {v.shape}")
