"""GAN-based style transfer model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Generator(nn.Module):
    """Generator for GAN-based style transfer."""
    
    def __init__(
        self,
        input_dim: int = 128,
        style_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        
        # Style embedding
        self.style_embedding = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Style injection layers
        self.style_injection = nn.ModuleList([
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 512, kernel_size=1)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Adaptive instance normalization
        self.adain_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 512 * 2) for _ in range(num_layers)
        ])
        
    def adaptive_instance_norm(
        self,
        x: torch.Tensor,
        style_params: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Apply adaptive instance normalization."""
        # Get style parameters
        style_params = self.adain_layers[layer_idx](style_params)
        gamma, beta = style_params.chunk(2, dim=1)
        
        # Reshape for broadcasting
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        
        # Compute mean and variance
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + 1e-8)
        
        # Apply style
        return gamma * x_norm + beta
    
    def forward(
        self,
        content_audio: torch.Tensor,
        style_audio: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through generator."""
        # Encode content
        content_features = self.content_encoder(content_audio)
        
        # Encode style
        style_features = self.style_embedding(style_audio.mean(dim=[2, 3]))
        
        # Apply style injection with AdaIN
        stylized_features = content_features
        for i, injection_layer in enumerate(self.style_injection):
            stylized_features = injection_layer(stylized_features)
            stylized_features = self.adaptive_instance_norm(
                stylized_features, style_features, i
            )
            stylized_features = F.relu(stylized_features)
        
        # Decode to output
        output = self.decoder(stylized_features)
        
        return output


class Discriminator(nn.Module):
    """Discriminator for GAN training."""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Convolutional layers
        layers = []
        in_channels = 1
        
        for i in range(num_layers):
            out_channels = min(hidden_dim * (2 ** i), 512)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        features = self.conv_layers(x)
        output = self.output_layer(features)
        return output


class GANStyleTransferModel(nn.Module):
    """GAN-based style transfer model."""
    
    def __init__(
        self,
        input_dim: int = 128,
        style_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        
        # Generator and discriminator
        self.generator = Generator(
            input_dim=input_dim,
            style_dim=style_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.discriminator = Discriminator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
    def forward(
        self,
        content_audio: torch.Tensor,
        style_audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        # Generate stylized audio
        stylized_features = self.generator(content_audio, style_audio)
        
        # Discriminator outputs
        real_output = self.discriminator(content_audio)
        fake_output = self.discriminator(stylized_features)
        
        return stylized_features, real_output, fake_output
    
    def transfer_style(
        self,
        content_audio: torch.Tensor,
        style_audio: torch.Tensor
    ) -> torch.Tensor:
        """Apply style transfer to content audio."""
        stylized_features, _, _ = self.forward(content_audio, style_audio)
        return stylized_features
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate between real and fake audio."""
        return self.discriminator(x)
