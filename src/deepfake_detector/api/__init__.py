"""
API module for deepfake detection backend
"""

from .inference_service import InferenceService
from .api_server import create_app

__all__ = ['InferenceService', 'create_app']
