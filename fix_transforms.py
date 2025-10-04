import sys
from pathlib import Path

# Read and fix the file
file_path = Path('train_production_grade.py')
content = file_path.read_text()

# Fix RandomResizedCrop
content = content.replace(
    'A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),',
    'A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0)),'
)

# Fix Resize
content = content.replace(
    'A.Resize(224, 224),',
    'A.Resize(height=224, width=224),'
)

file_path.write_text(content)
print("Fixed albumentations syntax")
