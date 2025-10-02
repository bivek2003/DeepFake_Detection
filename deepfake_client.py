#!/usr/bin/env python3
"""
Python client library for Deepfake Detection API
"""

import requests
from typing import Dict, Optional
from pathlib import Path

class DeepfakeClient:
    """Client for Deepfake Detection API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict_image(self, image_path: str) -> Dict:
        """Predict if image is deepfake"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                f"{self.base_url}/predict/image",
                files=files
            )
        response.raise_for_status()
        return response.json()
    
    def predict_video(self, video_path: str) -> Dict:
        """Predict if video is deepfake"""
        with open(video_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                f"{self.base_url}/predict/video",
                files=files
            )
        response.raise_for_status()
        return response.json()

# Example usage
if __name__ == "__main__":
    client = DeepfakeClient()
    
    # Health check
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # Predict video
    result = client.predict_video("test_video.mp4")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
