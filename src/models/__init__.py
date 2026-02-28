"""Model factory for creating different style transfer models."""

import torch
import torch.nn as nn
from typing import Dict, Type, Any

from .cnn_model import CNNStyleTransferModel
from .transformer_model import TransformerStyleTransferModel
from .gan_model import GANStyleTransferModel


class ModelFactory:
    """Factory class for creating style transfer models."""
    
    _models: Dict[str, Type[nn.Module]] = {
        'cnn': CNNStyleTransferModel,
        'transformer': TransformerStyleTransferModel,
        'gan': GANStyleTransferModel
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        **kwargs: Any
    ) -> nn.Module:
        """Create a style transfer model of the specified type."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {list(cls._models.keys())}")
        
        model_class = cls._models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types."""
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[nn.Module]) -> None:
        """Register a new model type."""
        cls._models[name] = model_class


class StyleTransferModel(nn.Module):
    """Main style transfer model wrapper."""
    
    def __init__(
        self,
        model_type: str = 'cnn',
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        **kwargs: Any
    ):
        super().__init__()
        
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Create the underlying model
        self.model = ModelFactory.create_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
        
    def forward(
        self,
        content_audio: torch.Tensor,
        style_audio: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model.transfer_style(content_audio, style_audio)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'model_info': self.get_model_info()
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: torch.device = None) -> 'StyleTransferModel':
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            model_type=checkpoint['model_type'],
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        return model
