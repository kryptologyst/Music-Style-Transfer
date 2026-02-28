"""Training utilities for music style transfer."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple
import os
import time
from tqdm import tqdm
import wandb
from pathlib import Path

from ..models import StyleTransferModel
from ..losses import CombinedLoss
from ..utils.device import get_device, save_checkpoint, load_checkpoint
from ..utils.config import Config


class StyleTransferTrainer:
    """Trainer for music style transfer models."""
    
    def __init__(
        self,
        model: StyleTransferModel,
        config: Config,
        train_loader,
        val_loader,
        device: Optional[torch.device] = None
    ):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device(config.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Initialize loss function
        self.criterion = CombinedLoss(
            style_weight=config.model.style_weight,
            content_weight=config.model.content_weight,
            perceptual_weight=config.model.perceptual_weight
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.learning_rate * 0.01
        )
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.__dict__
            )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_style_loss = 0.0
        total_content_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            content = batch['content'].to(self.device)
            style = batch['style'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            stylized = self.model(content, style)
            
            # Compute loss
            losses = self.criterion(
                generated_audio=stylized,
                content_features=content,
                style_features=style
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses['total'].item()
            total_style_loss += losses.get('style', 0).item()
            total_content_loss += losses.get('content', 0).item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'style': f"{losses.get('style', 0).item():.4f}",
                'content': f"{losses.get('content', 0).item():.4f}"
            })
        
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_style_loss = total_style_loss / num_batches
        avg_content_loss = total_content_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'train_style_loss': avg_style_loss,
            'train_content_loss': avg_content_loss
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_style_loss = 0.0
        total_content_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                content = batch['content'].to(self.device)
                style = batch['style'].to(self.device)
                
                # Forward pass
                stylized = self.model(content, style)
                
                # Compute loss
                losses = self.criterion(
                    generated_audio=stylized,
                    content_features=content,
                    style_features=style
                )
                
                # Update metrics
                total_loss += losses['total'].item()
                total_style_loss += losses.get('style', 0).item()
                total_content_loss += losses.get('content', 0).item()
                num_batches += 1
        
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_style_loss = total_style_loss / num_batches
        avg_content_loss = total_content_loss / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_style_loss': avg_style_loss,
            'val_content_loss': avg_content_loss
        }
    
    def train(self) -> None:
        """Train the model."""
        print(f"Starting training for {self.config.training.num_epochs} epochs...")
        print(f"Model: {self.model.get_model_info()}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            if epoch % self.config.training.save_interval == 0:
                self._save_checkpoint(epoch, val_metrics['val_loss'])
            
            # Early stopping
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.early_stopping_counter = 0
                self._save_checkpoint(epoch, val_metrics['val_loss'], is_best=True)
            else:
                self.early_stopping_counter += 1
            
            if self.early_stopping_counter >= self.config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print epoch summary
            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Close logging
        self.writer.close()
        if self.config.use_wandb:
            wandb.finish()
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics to tensorboard and wandb."""
        # Tensorboard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"Train/{key}", value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"Val/{key}", value, epoch)
        
        self.writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]['lr'], epoch)
        
        # Wandb
        if self.config.use_wandb:
            log_dict = {**train_metrics, **val_metrics}
            log_dict['epoch'] = epoch
            log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
            wandb.log(log_dict)
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest.pth"
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss,
            filepath=str(latest_path),
            metadata={
                'config': self.config.__dict__,
                'model_info': self.model.get_model_info()
            }
        )
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best.pth"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=loss,
                filepath=str(best_path),
                metadata={
                    'config': self.config.__dict__,
                    'model_info': self.model.get_model_info()
                }
            )
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = load_checkpoint(
            filepath=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            device=self.device
        )
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")