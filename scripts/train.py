#!/usr/bin/env python3
"""Training script for music style transfer."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from omegaconf import OmegaConf

from src.models import StyleTransferModel
from src.data import create_dataloaders
from src.training import StyleTransferTrainer
from src.utils.config import Config, load_config, get_default_config
from src.utils.device import get_device, set_seed


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train music style transfer model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "transformer", "gan"], help="Model type")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    config.data.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    config.model.model_type = args.model_type
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.num_epochs = args.num_epochs
    config.device = args.device
    config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed)
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config.data.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.num_workers,
        max_duration=config.audio.max_duration,
        augment=config.data.augment
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print(f"Creating {config.model.model_type} model...")
    model = StyleTransferModel(
        model_type=config.model.model_type,
        input_dim=config.audio.n_mels,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )
    
    print(f"Model info: {model.get_model_info()}")
    
    # Create trainer
    trainer = StyleTransferTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
