"""
FastAPI server for deepfake detection - Updated for robust model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
import logging
from pathlib import Path

from .inference_service import InferenceService

logger = logging.getLogger(__name__)

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    fake_probability: float
    real_probability: float
    frames_analyzed: Optional[int] = None
    total_frames: Optional[int] = None
    is_confident: bool
    warning: Optional[str] = None

def create_app(model_path: str = "./models/faceforensics_robust.pth"):
    """Create FastAPI application with robust model"""
    
    app = FastAPI(
        title="Deepfake Detection API - Robust Version",
        description="Production API with robust deepfake detection trained on diverse data",
        version="2.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize inference service with confidence threshold
    try:
        inference_service = InferenceService(model_path, confidence_threshold=0.65)
        logger.info("Robust inference service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    @app.get("/")
    async def root():
        return {
            "status": "healthy",
            "service": "Deepfake Detection API - Robust",
            "version": "2.0.0",
            "model": "Trained on 6 deepfake methods + heavy augmentation",
            "features": "Improved generalization to real-world videos"
        }
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "model_loaded": inference_service.model is not None,
            "device": str(inference_service.device),
            "confidence_threshold": inference_service.confidence_threshold
        }
    
    @app.post("/predict/image", response_model=PredictionResponse)
    async def predict_image(file: UploadFile = File(...)):
        """Predict if uploaded image is real or fake"""
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                contents = await file.read()
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            import cv2
            image = cv2.imread(tmp_path)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            result = inference_service.predict_image(image)
            os.unlink(tmp_path)
            
            return PredictionResponse(**result)
        
        except Exception as e:
            logger.error(f"Image prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/video", response_model=PredictionResponse)
    async def predict_video(file: UploadFile = File(...)):
        """Predict if uploaded video is real or fake"""
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                contents = await file.read()
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            result = inference_service.predict_video(tmp_path)
            os.unlink(tmp_path)
            
            return PredictionResponse(**result)
        
        except Exception as e:
            logger.error(f"Video prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
