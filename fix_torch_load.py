from pathlib import Path

# Fix inference_service.py
file1 = Path('src/deepfake_detector/api/inference_service.py')
content = file1.read_text()
content = content.replace(
    'checkpoint = torch.load(self.model_path, map_location=self.device)',
    'checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)'
)
file1.write_text(content)

# Fix infer_real_videos.py
file2 = Path('infer_real_videos.py')
if file2.exists():
    content = file2.read_text()
    content = content.replace(
        'checkpoint = torch.load(model_path, map_location=device)',
        'checkpoint = torch.load(model_path, map_location=device, weights_only=False)'
    )
    file2.write_text(content)

# Fix test_final_model.py
file3 = Path('test_final_model.py')
if file3.exists():
    content = file3.read_text()
    content = content.replace(
        'checkpoint = torch.load(\'models/faceforensics_improved.pth\', map_location=device)',
        'checkpoint = torch.load(\'models/faceforensics_improved.pth\', map_location=device, weights_only=False)'
    )
    file3.write_text(content)

print("Fixed all torch.load calls")
