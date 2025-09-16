"""
Model Registry System for Deepfake Detection

This module provides a centralized registry for all deepfake detection models,
enabling easy model discovery, loading, and metadata management.
"""

import logging
from typing import Dict, List, Optional, Type, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    model_class: Type[nn.Module]
    description: str
    input_type: str  # 'image', 'audio', 'video'
    input_shape: tuple
    num_classes: int = 2
    pretrained_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    memory_mb: Optional[int] = None
    inference_speed: Optional[str] = None  # 'fast', 'medium', 'slow'
    accuracy: Optional[float] = None
    
    def __post_init__(self):
        """Validate model info after initialization."""
        valid_input_types = {'image', 'audio', 'video', 'multimodal'}
        if self.input_type not in valid_input_types:
            raise ValueError(f"input_type must be one of {valid_input_types}")
        
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")

class ModelRegistry:
    """Central registry for deepfake detection models."""
    
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._aliases: Dict[str, str] = {}
        
    def register(self, model_info: ModelInfo) -> None:
        """Register a new model."""
        if model_info.name in self._models:
            logger.warning(f"Model {model_info.name} already registered. Overwriting.")
        
        self._models[model_info.name] = model_info
        logger.info(f"Registered model: {model_info.name}")
    
    def register_alias(self, alias: str, model_name: str) -> None:
        """Register an alias for a model."""
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        self._aliases[alias] = model_name
        logger.info(f"Registered alias: {alias} -> {model_name}")
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get model information by name or alias."""
        # Check direct name first
        if name in self._models:
            return self._models[name]
        
        # Check aliases
        if name in self._aliases:
            actual_name = self._aliases[name]
            return self._models[actual_name]
        
        return None
    
    def list_models(self, input_type: Optional[str] = None) -> List[str]:
        """List all registered models, optionally filtered by input type."""
        models = []
        for name, info in self._models.items():
            if input_type is None or info.input_type == input_type:
                models.append(name)
        return sorted(models)
    
    def get_models_by_type(self, input_type: str) -> Dict[str, ModelInfo]:
        """Get all models of a specific input type."""
        return {
            name: info for name, info in self._models.items() 
            if info.input_type == input_type
        }
    
    def create_model(self, name: str, **kwargs) -> nn.Module:
        """Create a model instance by name."""
        info = self.get_model_info(name)
        if info is None:
            available = ', '.join(self.list_models())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        
        try:
            # Create model instance
            model = info.model_class(**kwargs)
            logger.info(f"Created model: {name}")
            return model
        except Exception as e:
            logger.error(f"Failed to create model {name}: {e}")
            raise
    
    def unregister(self, name: str) -> bool:
        """Unregister a model."""
        if name in self._models:
            del self._models[name]
            # Remove any aliases pointing to this model
            aliases_to_remove = [alias for alias, model in self._aliases.items() if model == name]
            for alias in aliases_to_remove:
                del self._aliases[alias]
            logger.info(f"Unregistered model: {name}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all registered models."""
        self._models.clear()
        self._aliases.clear()
        logger.info("Cleared all registered models")
    
    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self._models)
    
    def __contains__(self, name: str) -> bool:
        """Check if model is registered."""
        return name in self._models or name in self._aliases
    
    def __iter__(self):
        """Iterate over model names."""
        return iter(self._models.keys())

# Global registry instance
model_registry = ModelRegistry()

def register_model(name: str, model_class: Type[nn.Module], **kwargs) -> None:
    """Convenience function to register a model."""
    info = ModelInfo(name=name, model_class=model_class, **kwargs)
    model_registry.register(info)

def get_model_info(name: str) -> Optional[ModelInfo]:
    """Get model information."""
    return model_registry.get_model_info(name)

def list_models(input_type: Optional[str] = None) -> List[str]:
    """List available models."""
    return model_registry.list_models(input_type)

def create_model(name: str, **kwargs) -> nn.Module:
    """Create a model instance."""
    return model_registry.create_model(name, **kwargs)

def get_model(name: str, **kwargs) -> nn.Module:
    """Alias for create_model - for backward compatibility."""
    return create_model(name, **kwargs)

# Decorator for automatic model registration
def register_deepfake_model(name: str, **model_info_kwargs):
    """Decorator to automatically register a model class."""
    def decorator(model_class: Type[nn.Module]):
        info = ModelInfo(name=name, model_class=model_class, **model_info_kwargs)
        model_registry.register(info)
        return model_class
    return decorator

def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return model_registry

# Export all necessary items
__all__ = [
    'ModelInfo',
    'ModelRegistry', 
    'model_registry',
    'register_model',
    'get_model_info',
    'list_models',
    'create_model',
    'get_model',  # Added for backward compatibility
    'register_deepfake_model',
    'get_registry'
]
