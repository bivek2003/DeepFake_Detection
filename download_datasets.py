#!/usr/bin/env python3
"""
Download helper for deepfake datasets
"""

import os
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_instructions():
    """Print download instructions for datasets"""
    
    logger.info("DEEPFAKE DATASET DOWNLOAD GUIDE")
    logger.info("="*50)
    
    logger.info("\n1. FACEFORENSICS++ (Primary - 150GB compressed)")
    logger.info("   - Go to: https://github.com/ondyari/FaceForensics")
    logger.info("   - Fill out the form: https://docs.google.com/forms/...")
    logger.info("   - Request: c23 compressed version")
    logger.info("   - Extract to: ./datasets/faceforensics/")
    logger.info("   - Expected structure:")
    logger.info("     ./datasets/faceforensics/")
    logger.info("     ├── original_sequences/youtube/c23/videos/")
    logger.info("     └── manipulated_sequences/[Deepfakes,FaceSwap,Face2Face,NeuralTextures]/c23/videos/")
    
    logger.info("\n2. CELEB-DF (Secondary - 80GB)")
    logger.info("   - Go to: https://github.com/yuezunli/celeb-deepfakeforensics")
    logger.info("   - Fill out the form in repository")
    logger.info("   - Extract to: ./datasets/celebdf/")
    logger.info("   - Expected structure:")
    logger.info("     ./datasets/celebdf/")
    logger.info("     ├── Celeb-real/")
    logger.info("     └── Celeb-synthesis/")
    
    logger.info("\n3. KAGGLE ALTERNATIVE (If above unavailable)")
    logger.info("   - Search 'deepfake detection' on Kaggle")
    logger.info("   - Download smaller subsets for testing")
    
    logger.info("\nAfter download, run:")
    logger.info("python train_real_deepfakes.py --faceforensics_path ./datasets/faceforensics --celebdf_path ./datasets/celebdf")

if __name__ == "__main__":
    download_instructions()
