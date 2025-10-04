#!/usr/bin/env python3
import uvicorn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from deepfake_detector.api.api_server import create_app

if __name__ == "__main__":
    model_path = "./models/faceforensics_improved.pth"  # Use original 94% model
    app = create_app(model_path=model_path)
    
    print("="*60)
    print("Deepfake Detection API - Original Model (94% accuracy)")
    print("="*60)
    print("API: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
