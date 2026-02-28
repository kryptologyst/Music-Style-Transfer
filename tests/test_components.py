"""Tests for music style transfer components."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import StyleTransferModel, ModelFactory
from features.audio_processor import AudioProcessor
from losses import CombinedLoss, StyleLoss, ContentLoss
from metrics import StyleTransferMetrics
from utils.device import get_device, set_seed
from utils.config import Config, get_default_config


class TestAudioProcessor:
    """Test audio processor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
        self.sample_rate = 22050
        self.duration = 2.0
        self.samples = int(self.duration * self.sample_rate)
        
        # Create synthetic audio
        self.audio = np.sin(2 * np.pi * 440 * np.linspace(0, self.duration, self.samples))
    
    def test_load_save_audio(self):
        """Test audio loading and saving."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Save audio
            self.processor.save_audio(self.audio, tmp_file.name, self.sample_rate)
            
            # Load audio
            loaded_audio, loaded_sr = self.processor.load_audio(tmp_file.name)
            
            # Clean up
            os.unlink(tmp_file.name)
        
        assert loaded_sr == self.sample_rate
        assert len(loaded_audio) == len(self.audio)
        assert np.allclose(loaded_audio, self.audio, atol=1e-6)
    
    def test_feature_extraction(self):
        """Test feature extraction methods."""
        # Test mel spectrogram
        mel_spec = self.processor.extract_mel_spectrogram(self.audio)
        assert mel_spec.shape[0] == self.processor.n_mels
        assert mel_spec.shape[1] > 0
        
        # Test MFCC
        mfcc = self.processor.extract_mfcc(self.audio)
        assert mfcc.shape[0] == self.processor.n_mfcc
        assert mfcc.shape[1] > 0
        
        # Test chroma
        chroma = self.processor.extract_chroma(self.audio)
        assert chroma.shape[0] == 12  # 12 chroma bins
        assert chroma.shape[1] > 0
    
    def test_preprocessing(self):
        """Test audio preprocessing."""
        processed_audio = self.processor.preprocess_audio(self.audio)
        
        assert len(processed_audio) == len(self.audio)
        assert not np.allclose(processed_audio, self.audio)  # Should be different after preprocessing
    
    def test_pad_or_truncate(self):
        """Test padding and truncation."""
        # Test truncation
        short_audio = self.processor.pad_or_truncate(self.audio, len(self.audio) // 2)
        assert len(short_audio) == len(self.audio) // 2
        
        # Test padding
        long_audio = self.processor.pad_or_truncate(self.audio, len(self.audio) * 2)
        assert len(long_audio) == len(self.audio) * 2
        assert np.allclose(long_audio[:len(self.audio)], self.audio)


class TestModels:
    """Test model functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.input_dim = 128
        self.time_frames = 100
        self.content_input = torch.randn(self.batch_size, 1, self.input_dim, self.time_frames)
        self.style_input = torch.randn(self.batch_size, 1, self.input_dim, self.time_frames)
    
    def test_cnn_model(self):
        """Test CNN model."""
        model = ModelFactory.create_model('cnn', input_dim=self.input_dim)
        
        # Test forward pass
        stylized = model.transfer_style(self.content_input, self.style_input)
        
        assert stylized.shape == self.content_input.shape
        assert not torch.isnan(stylized).any()
        assert not torch.isinf(stylized).any()
    
    def test_transformer_model(self):
        """Test Transformer model."""
        model = ModelFactory.create_model('transformer', input_dim=self.input_dim)
        
        # Test forward pass
        stylized = model.transfer_style(self.content_input, self.style_input)
        
        assert stylized.shape == self.content_input.shape
        assert not torch.isnan(stylized).any()
        assert not torch.isinf(stylized).any()
    
    def test_gan_model(self):
        """Test GAN model."""
        model = ModelFactory.create_model('gan', input_dim=self.input_dim)
        
        # Test forward pass
        stylized = model.transfer_style(self.content_input, self.style_input)
        
        assert stylized.shape == self.content_input.shape
        assert not torch.isnan(stylized).any()
        assert not torch.isinf(stylized).any()
    
    def test_model_factory(self):
        """Test model factory."""
        # Test available models
        available_models = ModelFactory.get_available_models()
        assert 'cnn' in available_models
        assert 'transformer' in available_models
        assert 'gan' in available_models
        
        # Test invalid model type
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model')
    
    def test_style_transfer_model_wrapper(self):
        """Test StyleTransferModel wrapper."""
        model = StyleTransferModel(model_type='cnn', input_dim=self.input_dim)
        
        # Test forward pass
        stylized = model(self.content_input, self.style_input)
        
        assert stylized.shape == self.content_input.shape
        
        # Test model info
        info = model.get_model_info()
        assert 'model_type' in info
        assert 'total_parameters' in info
        assert info['model_type'] == 'cnn'


class TestLosses:
    """Test loss functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.input_dim = 128
        self.time_frames = 100
        self.generated = torch.randn(self.batch_size, 1, self.input_dim, self.time_frames)
        self.content = torch.randn(self.batch_size, 1, self.input_dim, self.time_frames)
        self.style = torch.randn(self.batch_size, 1, self.input_dim, self.time_frames)
    
    def test_style_loss(self):
        """Test style loss."""
        loss_fn = StyleLoss()
        loss = loss_fn(self.generated, self.style)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_content_loss(self):
        """Test content loss."""
        loss_fn = ContentLoss()
        loss = loss_fn(self.generated, self.content)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_combined_loss(self):
        """Test combined loss."""
        loss_fn = CombinedLoss()
        losses = loss_fn(self.generated, self.content, self.style)
        
        assert isinstance(losses, dict)
        assert 'total' in losses
        assert 'style' in losses
        assert 'content' in losses
        
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.item() >= 0
            assert not torch.isnan(loss_value)


class TestMetrics:
    """Test evaluation metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
        self.metrics_calculator = StyleTransferMetrics(self.processor)
        
        # Create synthetic audio
        self.sample_rate = 22050
        self.duration = 2.0
        self.samples = int(self.duration * self.sample_rate)
        
        self.content_audio = np.sin(2 * np.pi * 440 * np.linspace(0, self.duration, self.samples))
        self.style_audio = np.sin(2 * np.pi * 880 * np.linspace(0, self.duration, self.samples))
        self.generated_audio = np.sin(2 * np.pi * 660 * np.linspace(0, self.duration, self.samples))
    
    def test_style_similarity(self):
        """Test style similarity metric."""
        similarity = self.metrics_calculator.compute_style_similarity(
            self.generated_audio, self.style_audio
        )
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_content_preservation(self):
        """Test content preservation metric."""
        preservation = self.metrics_calculator.compute_content_preservation(
            self.generated_audio, self.content_audio
        )
        
        assert isinstance(preservation, float)
        assert 0.0 <= preservation <= 1.0
    
    def test_spectral_distance(self):
        """Test spectral distance metric."""
        distance = self.metrics_calculator.compute_spectral_distance(
            self.generated_audio, self.content_audio
        )
        
        assert isinstance(distance, float)
        assert distance >= 0.0
    
    def test_all_metrics(self):
        """Test all metrics computation."""
        metrics = self.metrics_calculator.compute_all_metrics(
            self.generated_audio, self.content_audio, self.style_audio
        )
        
        expected_metrics = [
            'style_similarity', 'content_preservation', 'spectral_distance',
            'rhythm_similarity', 'harmonicity', 'overall_quality'
        ]
        
        for metric_name in expected_metrics:
            assert metric_name in metrics
            assert isinstance(metrics[metric_name], float)
            assert not np.isnan(metrics[metric_name])


class TestUtils:
    """Test utility functions."""
    
    def test_device_selection(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        rand1 = torch.randn(10)
        set_seed(42)
        rand2 = torch.randn(10)
        
        # Should be the same with same seed
        assert torch.allclose(rand1, rand2)
    
    def test_config(self):
        """Test configuration management."""
        config = get_default_config()
        
        assert isinstance(config, Config)
        assert hasattr(config, 'audio')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'data')


if __name__ == "__main__":
    pytest.main([__file__])
