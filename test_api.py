#!/usr/bin/env python3
"""
Comprehensive API testing suite
"""

import requests
import json
from pathlib import Path
import time

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Health check passed")

def test_image_prediction(image_path):
    """Test image prediction endpoint"""
    print("\n" + "="*60)
    print(f"Testing Image Prediction: {image_path}")
    print("="*60)
    
    with open(image_path, 'rb') as f:
        files = {'file': ('image.jpg', f, 'image/jpeg')}
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/predict/image", files=files)
        elapsed = time.time() - start_time
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Fake Probability: {result['fake_probability']:.3f}")
        print(f"Real Probability: {result['real_probability']:.3f}")
        print("✓ Image prediction passed")
    else:
        print(f"✗ Error: {response.text}")

def test_video_prediction(video_path):
    """Test video prediction endpoint"""
    print("\n" + "="*60)
    print(f"Testing Video Prediction: {video_path}")
    print("="*60)
    
    with open(video_path, 'rb') as f:
        files = {'file': ('video.mp4', f, 'video/mp4')}
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/predict/video", files=files)
        elapsed = time.time() - start_time
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Frames Analyzed: {result['frames_analyzed']}/{result['total_frames']}")
        print(f"Fake Probability: {result['fake_probability']:.3f}")
        print("✓ Video prediction passed")
    else:
        print(f"✗ Error: {response.text}")

def test_performance_benchmark(video_path, num_requests=10):
    """Benchmark API performance"""
    print("\n" + "="*60)
    print(f"Performance Benchmark ({num_requests} requests)")
    print("="*60)
    
    times = []
    for i in range(num_requests):
        with open(video_path, 'rb') as f:
            files = {'file': ('video.mp4', f, 'video/mp4')}
            start = time.time()
            response = requests.post(f"{API_BASE_URL}/predict/video", files=files)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"Request {i+1}/{num_requests}: {elapsed:.2f}s")
    
    print(f"\nAverage: {sum(times)/len(times):.2f}s")
    print(f"Min: {min(times):.2f}s")
    print(f"Max: {max(times):.2f}s")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Test image path')
    parser.add_argument('--video', type=str, help='Test video path')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    args = parser.parse_args()
    
    print("Deepfake Detection API - Test Suite")
    
    # Test health
    test_health_check()
    
    # Test image if provided
    if args.image:
        test_image_prediction(args.image)
    
    # Test video if provided
    if args.video:
        test_video_prediction(args.video)
        
        if args.benchmark:
            test_performance_benchmark(args.video, num_requests=5)
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
