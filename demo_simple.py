#!/usr/bin/env python3
"""Simple demo script for music style transfer."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from src.models import StyleTransferModel
from src.features.audio_processor import AudioProcessor
from src.utils.device import get_device, set_seed


def main():
    """Run a simple demo of music style transfer."""
    print("🎵 Music Style Transfer Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device("auto")
    print(f"Using device: {device}")
    
    # Initialize components
    processor = AudioProcessor()
    model = StyleTransferModel(model_type="cnn")
    model = model.to(device)
    
    print(f"Model info: {model.get_model_info()}")
    
    # Create synthetic audio data
    print("\nCreating synthetic audio data...")
    sample_rate = 22050
    duration = 2.0
    samples = int(duration * sample_rate)
    
    # Content audio (sine wave at 440 Hz)
    content_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
    
    # Style audio (sine wave at 880 Hz with harmonics)
    style_audio = (np.sin(2 * np.pi * 880 * np.linspace(0, duration, samples)) +
                   np.sin(2 * np.pi * 1760 * np.linspace(0, duration, samples)) * 0.5)
    
    print(f"Content audio: {len(content_audio)} samples")
    print(f"Style audio: {len(style_audio)} samples")
    
    # Preprocess audio
    content_audio = processor.preprocess_audio(content_audio)
    style_audio = processor.preprocess_audio(style_audio)
    
    # Extract features
    print("\nExtracting features...")
    content_features = processor.extract_mel_spectrogram(content_audio)
    style_features = processor.extract_mel_spectrogram(style_audio)
    
    print(f"Content features shape: {content_features.shape}")
    print(f"Style features shape: {style_features.shape}")
    
    # Convert to tensors
    content_tensor = torch.from_numpy(content_features).float().unsqueeze(0).unsqueeze(0)
    style_tensor = torch.from_numpy(style_features).float().unsqueeze(0).unsqueeze(0)
    
    # Move to device
    content_tensor = content_tensor.to(device)
    style_tensor = style_tensor.to(device)
    
    # Apply style transfer
    print("\nApplying style transfer...")
    with torch.no_grad():
        stylized_features = model(content_tensor, style_tensor)
    
    print(f"Stylized features shape: {stylized_features.shape}")
    
    # Convert back to numpy
    stylized_audio = stylized_features.squeeze().cpu().numpy()
    
    # Save results
    print("\nSaving results...")
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    processor.save_audio(content_audio, output_dir / "content.wav", sample_rate)
    processor.save_audio(style_audio, output_dir / "style.wav", sample_rate)
    processor.save_audio(stylized_audio, output_dir / "stylized.wav", sample_rate)
    
    print(f"Results saved to {output_dir}/")
    print("- content.wav: Original content audio")
    print("- style.wav: Style reference audio")
    print("- stylized.wav: Style-transferred audio")
    
    # Compute basic metrics
    print("\nComputing basic metrics...")
    from src.metrics import StyleTransferMetrics
    metrics_calculator = StyleTransferMetrics(processor)
    
    metrics = metrics_calculator.compute_all_metrics(
        stylized_audio, content_audio, style_audio
    )
    
    print("\nEvaluation Results:")
    print("-" * 30)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Quality assessment
    overall_score = metrics['overall_quality']
    if overall_score >= 0.8:
        print("\n✅ Excellent: High-quality style transfer!")
    elif overall_score >= 0.6:
        print("\n✅ Good: Decent style transfer capabilities.")
    elif overall_score >= 0.4:
        print("\n⚠️ Fair: Basic functionality, needs improvement.")
    else:
        print("\n❌ Poor: Requires significant improvements.")
    
    print("\n🎉 Demo completed successfully!")
    print("\nTo run the interactive demo:")
    print("streamlit run demo/streamlit_app.py")


if __name__ == "__main__":
    main()
