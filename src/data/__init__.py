"""Data loading and preprocessing for music style transfer."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import random

from ..features.audio_processor import AudioProcessor


class MusicStyleDataset(Dataset):
    """Dataset for music style transfer."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_duration: float = 30.0,
        augment: bool = True,
        processor: Optional[AudioProcessor] = None
    ):
        """Initialize dataset."""
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_duration = max_duration
        self.augment = augment
        
        # Initialize audio processor
        if processor is None:
            self.processor = AudioProcessor()
        else:
            self.processor = processor
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter by split
        self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        
        # Create pairs for style transfer
        self.pairs = self._create_pairs()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata."""
        meta_path = self.data_dir / 'meta.csv'
        
        if meta_path.exists():
            return pd.read_csv(meta_path)
        else:
            # Create synthetic metadata if not exists
            return self._create_synthetic_metadata()
    
    def _create_synthetic_metadata(self) -> pd.DataFrame:
        """Create synthetic metadata for demonstration."""
        # Create synthetic data directory structure
        content_dir = self.data_dir / 'raw' / 'content'
        style_dir = self.data_dir / 'raw' / 'style'
        
        content_dir.mkdir(parents=True, exist_ok=True)
        style_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic audio files
        metadata = []
        splits = ['train', 'val', 'test']
        
        for split in splits:
            for i in range(10):  # 10 files per split
                # Generate synthetic audio
                duration = random.uniform(5, 15)
                sample_rate = 22050
                samples = int(duration * sample_rate)
                
                # Create different types of synthetic audio
                if i % 3 == 0:
                    # Sine wave
                    freq = random.uniform(200, 800)
                    audio = np.sin(2 * np.pi * freq * np.linspace(0, duration, samples))
                elif i % 3 == 1:
                    # Noise
                    audio = np.random.normal(0, 0.1, samples)
                else:
                    # Mixed signal
                    freq1 = random.uniform(200, 400)
                    freq2 = random.uniform(600, 1000)
                    audio = (np.sin(2 * np.pi * freq1 * np.linspace(0, duration, samples)) +
                           np.sin(2 * np.pi * freq2 * np.linspace(0, duration, samples))) * 0.5
                
                # Save audio file
                file_path = content_dir / f"{split}_{i}.wav"
                self.processor.save_audio(audio, file_path, sample_rate)
                
                # Add to metadata
                metadata.append({
                    'id': f"{split}_{i}",
                    'path': str(file_path.relative_to(self.data_dir)),
                    'sr': sample_rate,
                    'duration': duration,
                    'genre': random.choice(['classical', 'jazz', 'rock', 'electronic']),
                    'split': split,
                    'type': 'content'
                })
        
        # Create style files (similar process)
        for split in splits:
            for i in range(5):  # 5 style files per split
                duration = random.uniform(5, 15)
                sample_rate = 22050
                samples = int(duration * sample_rate)
                
                # Create different style characteristics
                if i % 2 == 0:
                    # Harmonic content
                    audio = np.zeros(samples)
                    for harmonic in range(1, 6):
                        freq = random.uniform(100, 300) * harmonic
                        audio += np.sin(2 * np.pi * freq * np.linspace(0, duration, samples)) / harmonic
                else:
                    # Percussive content
                    audio = np.random.normal(0, 0.2, samples)
                    # Add some rhythmic patterns
                    beat_length = int(sample_rate * 0.5)  # 0.5 second beats
                    for beat in range(0, samples, beat_length):
                        if beat + beat_length < samples:
                            audio[beat:beat+beat_length//4] *= 2
                
                # Save audio file
                file_path = style_dir / f"{split}_style_{i}.wav"
                self.processor.save_audio(audio, file_path, sample_rate)
                
                # Add to metadata
                metadata.append({
                    'id': f"{split}_style_{i}",
                    'path': str(file_path.relative_to(self.data_dir)),
                    'sr': sample_rate,
                    'duration': duration,
                    'genre': random.choice(['classical', 'jazz', 'rock', 'electronic']),
                    'split': split,
                    'type': 'style'
                })
        
        # Save metadata
        df = pd.DataFrame(metadata)
        df.to_csv(self.data_dir / 'meta.csv', index=False)
        
        return df
    
    def _create_pairs(self) -> List[Dict[str, str]]:
        """Create content-style pairs for training."""
        content_files = self.metadata[self.metadata['type'] == 'content']
        style_files = self.metadata[self.metadata['type'] == 'style']
        
        pairs = []
        for _, content_row in content_files.iterrows():
            for _, style_row in style_files.iterrows():
                pairs.append({
                    'content_id': content_row['id'],
                    'content_path': content_row['path'],
                    'style_id': style_row['id'],
                    'style_path': style_row['path'],
                    'content_genre': content_row['genre'],
                    'style_genre': style_row['genre']
                })
        
        return pairs
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset."""
        pair = self.pairs[idx]
        
        # Load content and style audio
        content_path = self.data_dir / pair['content_path']
        style_path = self.data_dir / pair['style_path']
        
        content_audio, content_sr = self.processor.load_audio(content_path)
        style_audio, style_sr = self.processor.load_audio(style_path)
        
        # Preprocess audio
        content_audio = self.processor.preprocess_audio(content_audio)
        style_audio = self.processor.preprocess_audio(style_audio)
        
        # Truncate or pad to max duration
        max_samples = int(self.max_duration * self.processor.sample_rate)
        content_audio = self.processor.pad_or_truncate(content_audio, max_samples)
        style_audio = self.processor.pad_or_truncate(style_audio, max_samples)
        
        # Extract features
        content_features = self.processor.extract_mel_spectrogram(content_audio)
        style_features = self.processor.extract_mel_spectrogram(style_audio)
        
        # Data augmentation
        if self.augment and self.split == 'train':
            content_audio = self.processor.augment_audio(content_audio, content_sr)
            style_audio = self.processor.augment_audio(style_audio, style_sr)
            
            # Re-extract features after augmentation
            content_features = self.processor.extract_mel_spectrogram(content_audio)
            style_features = self.processor.extract_mel_spectrogram(style_audio)
        
        # Convert to tensors
        content_tensor = torch.from_numpy(content_features).float().unsqueeze(0)
        style_tensor = torch.from_numpy(style_features).float().unsqueeze(0)
        
        return {
            'content': content_tensor,
            'style': style_tensor,
            'content_id': pair['content_id'],
            'style_id': pair['style_id'],
            'content_genre': pair['content_genre'],
            'style_genre': pair['style_genre']
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    max_duration: float = 30.0,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets."""
    
    # Create datasets
    train_dataset = MusicStyleDataset(
        data_dir=data_dir,
        split='train',
        max_duration=max_duration,
        augment=augment
    )
    
    val_dataset = MusicStyleDataset(
        data_dir=data_dir,
        split='val',
        max_duration=max_duration,
        augment=False
    )
    
    test_dataset = MusicStyleDataset(
        data_dir=data_dir,
        split='test',
        max_duration=max_duration,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
