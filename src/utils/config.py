"""Configuration management for music style transfer."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from omegaconf import OmegaConf
import yaml
import os


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 13
    fmin: float = 0.0
    fmax: Optional[float] = None
    preemphasis: float = 0.97
    max_duration: float = 30.0  # seconds


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "cnn"  # cnn, transformer, gan
    input_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    attention_heads: int = 8
    style_weight: float = 1.0
    content_weight: float = 1.0
    perceptual_weight: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 10
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    save_interval: int = 10
    eval_interval: int = 5
    early_stopping_patience: int = 20


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    augment: bool = True
    noise_prob: float = 0.3
    pitch_shift_prob: float = 0.3
    time_stretch_prob: float = 0.3


@dataclass
class Config:
    """Main configuration class."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # System settings
    device: str = "auto"  # auto, cpu, cuda, mps
    num_workers: int = 4
    seed: int = 42
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "music-style-transfer"
    wandb_entity: Optional[str] = None


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.structured(Config(**config_dict))


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    config_dict = OmegaConf.to_yaml(config)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def update_config(config: Config, updates: Dict[str, Any]) -> Config:
    """Update configuration with new values."""
    config_dict = OmegaConf.to_yaml(config)
    config_dict.update(updates)
    return OmegaConf.structured(Config(**config_dict))
