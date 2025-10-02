"""
FastAPI server for deepfake detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
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

def create_app(model_path: str = "./models/faceforensics_improved.pth"):
    """Create FastAPI application"""
    
    app = FastAPI(
        title="Deepfake Detection API",
        description="Production API for detecting deepfake videos and images",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize inference service
    try:
        inference_service = InferenceService(model_path)
        logger.info("Inference service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize inference service: {e}")
        raise
    
    @app.get("/")
    async def root():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "Deepfake Detection API",
            "version": "1.0.0",
            "model_accuracy": "94.06%"
        }
    
    @app.get("/health")
    async def health_check():
        """Detailed health check"""
        return {
            "status": "healthy",
            "model_loaded": inference_service.model is not None,
            "device": str(inference_service.device)
        }
    
    @app.post("/predict/image", response_model=PredictionResponse)
    async def predict_image(file: UploadFile = File(...)):
        """
        Predict if uploaded image is real or fake
        """
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                contents = await file.read()
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            # Read image
            import cv2
            image = cv2.imread(tmp_path)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Predict
            result = inference_service.predict_image(image)
            
            # Cleanup
            os.unlink(tmp_path)
            
            return PredictionResponse(**result)
        
        except Exception as e:
            logger.error(f"Image prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/video", response_model=PredictionResponse)
    async def predict_video(file: UploadFile = File(...)):
        """
        Predict if uploaded video is real or fake
        """
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                contents = await file.read()
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            # Predict
            result = inference_service.predict_video(tmp_path)
            
            # Cleanup
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
