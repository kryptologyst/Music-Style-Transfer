"""CNN-based style transfer model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ContentEncoder(nn.Module):
    """Content encoder for extracting content features."""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv2d(1, 64, kernel_size=3, padding=1)
        )
        self.conv_layers.append(
            nn.Conv2d(64, 128, kernel_size=3, padding=1)
        )
        self.conv_layers.append(
            nn.Conv2d(128, 256, kernel_size=3, padding=1)
        )
        self.conv_layers.append(
            nn.Conv2d(256, 512, kernel_size=3, padding=1)
        )
        
        # Batch normalization
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(512)
        ])
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size
        self.flatten_size = self._get_flatten_size()
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.flatten_size, hidden_dim))
        self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
    def _get_flatten_size(self) -> int:
        """Calculate the size after flattening."""
        # Simulate forward pass to get size
        x = torch.randn(1, 1, self.input_dim, 100)  # Assume 100 time frames
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(self.bn_layers[i](conv(x)))
            x = self.pool(x)
        return x.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through content encoder."""
        # x shape: (batch_size, 1, n_mels, time_frames)
        
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(self.bn_layers[i](conv(x)))
            x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc in self.fc_layers[:-1]:
            x = F.relu(fc(x))
            x = self.dropout(x)
        
        x = self.fc_layers[-1](x)
        
        return x


class StyleEncoder(nn.Module):
    """Style encoder for extracting style features."""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv2d(1, 64, kernel_size=3, padding=1)
        )
        self.conv_layers.append(
            nn.Conv2d(64, 128, kernel_size=3, padding=1)
        )
        self.conv_layers.append(
            nn.Conv2d(128, 256, kernel_size=3, padding=1)
        )
        self.conv_layers.append(
            nn.Conv2d(256, 512, kernel_size=3, padding=1)
        )
        
        # Batch normalization
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(512)
        ])
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size
        self.flatten_size = self._get_flatten_size()
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.flatten_size, hidden_dim))
        self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
    def _get_flatten_size(self) -> int:
        """Calculate the size after flattening."""
        # Simulate forward pass to get size
        x = torch.randn(1, 1, self.input_dim, 100)  # Assume 100 time frames
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(self.bn_layers[i](conv(x)))
            x = self.pool(x)
        return x.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through style encoder."""
        # x shape: (batch_size, 1, n_mels, time_frames)
        
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(self.bn_layers[i](conv(x)))
            x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc in self.fc_layers[:-1]:
            x = F.relu(fc(x))
            x = self.dropout(x)
        
        x = self.fc_layers[-1](x)
        
        return x


class StyleTransferDecoder(nn.Module):
    """Decoder for generating stylized audio features."""
    
    def __init__(
        self,
        content_dim: int = 256,
        style_dim: int = 256,
        output_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Fusion layer
        self.fusion = nn.Linear(content_dim + style_dim, hidden_dim)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, output_dim)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, content_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder."""
        # Concatenate content and style features
        combined = torch.cat([content_features, style_features], dim=1)
        
        # Fusion
        x = F.relu(self.fusion(combined))
        x = self.dropout(x)
        
        # Decoder layers
        for layer in self.decoder_layers[:-1]:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        # Output layer
        x = self.decoder_layers[-1](x)
        
        return x


class CNNStyleTransferModel(nn.Module):
    """CNN-based style transfer model."""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoders
        self.content_encoder = ContentEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.style_encoder = StyleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = StyleTransferDecoder(
            content_dim=hidden_dim,
            style_dim=hidden_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout
        )
        
    def forward(
        self,
        content_audio: torch.Tensor,
        style_audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        # Encode content and style
        content_features = self.content_encoder(content_audio)
        style_features = self.style_encoder(style_audio)
        
        # Generate stylized features
        stylized_features = self.decoder(content_features, style_features)
        
        return stylized_features, content_features, style_features
    
    def transfer_style(
        self,
        content_audio: torch.Tensor,
        style_audio: torch.Tensor
    ) -> torch.Tensor:
        """Apply style transfer to content audio."""
        stylized_features, _, _ = self.forward(content_audio, style_audio)
        return stylized_features
