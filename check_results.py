import torch
import json

# Load the trained model checkpoint
checkpoint = torch.load('models/faceforensics_improved.pth', map_location='cpu')

print("="*60)
print("PHASE 2 - PRODUCTION MODEL RESULTS")
print("="*60)
print(f"Validation Accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")
print(f"F1 Score: {checkpoint.get('f1', 'N/A'):.4f}")
print(f"Training completed at epoch: {checkpoint.get('epoch', 'N/A')}")
print("\nModel ready for deployment!")
print("="*60)
