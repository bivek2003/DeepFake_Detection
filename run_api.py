#!/usr/bin/env python3
"""
Run the Deepfake Detection API server
"""

import uvicorn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from deepfake_detector.api.api_server import create_app

if __name__ == "__main__":
    app = create_app(model_path="./models/faceforensics_improved.pth")
    
    print("="*60)
    print("Deepfake Detection API Server")
    print("="*60)
    print("API running at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
