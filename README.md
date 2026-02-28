# Music Style Transfer

## PRIVACY DISCLAIMER

**IMPORTANT: This is a research and educational demonstration project. This software is NOT intended for production use, biometric identification, or voice cloning applications. Any misuse of this technology for creating deepfakes, impersonation, or unauthorized voice synthesis is strictly prohibited and may violate laws and ethical guidelines.**

## Overview

This project implements music style transfer using neural networks, allowing you to transfer the style (tempo, instrument arrangement, timbre) of one music piece to another while preserving the content (melody, harmony). This is a research demonstration focused on audio understanding and music information retrieval (MIR).

## Features

- **Multiple Model Architectures**: CNN, Transformer, and GAN-based style transfer models
- **Advanced Feature Extraction**: MFCC, Chroma, Spectral features, and learned representations
- **Interactive Demo**: Streamlit/Gradio web interface for real-time style transfer
- **Comprehensive Evaluation**: Style similarity, content preservation, and perceptual metrics
- **Modern PyTorch Implementation**: GPU acceleration with automatic device detection

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Music-Style-Transfer.git
cd Music-Style-Transfer

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models import StyleTransferModel
from src.data import AudioProcessor

# Load and process audio
processor = AudioProcessor()
content_audio, sr = processor.load_audio("path/to/content.wav")
style_audio, _ = processor.load_audio("path/to/style.wav")

# Initialize model
model = StyleTransferModel()

# Apply style transfer
stylized_audio = model.transfer_style(content_audio, style_audio)

# Save result
processor.save_audio(stylized_audio, "output.wav", sr)
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/streamlit_app.py

# Or launch Gradio demo
python demo/gradio_app.py
```

## Project Structure

```
music-style-transfer/
├── src/                    # Source code
│   ├── models/            # Neural network models
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature extraction
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation scripts
│   └── utils/             # Utility functions
├── data/                  # Data directory
│   ├── raw/              # Raw audio files
│   ├── processed/        # Processed features
│   └── meta.csv          # Dataset metadata
├── configs/              # Configuration files
├── scripts/              # Training and evaluation scripts
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── assets/               # Generated outputs and visualizations
├── demo/                 # Interactive demos
└── docs/                 # Documentation
```

## Dataset Schema

The project expects audio files in the following structure:

```
data/
├── raw/
│   ├── content/          # Content audio files
│   └── style/            # Style reference files
└── meta.csv              # Metadata with columns:
    - id: unique identifier
    - path: relative path to audio file
    - sr: sample rate
    - duration: audio duration in seconds
    - genre: music genre
    - split: train/val/test
```

## Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom config
python scripts/train.py --config configs/custom_config.yaml

# Resume training from checkpoint
python scripts/train.py --resume checkpoints/latest.pth
```

## Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/best.pth

# Generate style transfer examples
python scripts/generate_examples.py --checkpoint checkpoints/best.pth
```

## Models

### CNN-based Style Transfer
- Convolutional layers for feature extraction
- Content-style feature fusion
- Reconstruction decoder

### Transformer-based Style Transfer
- Self-attention mechanisms
- Cross-attention for style-content interaction
- Positional encoding for temporal relationships

### GAN-based Style Transfer
- Generator for style transfer
- Discriminator for style authenticity
- Adversarial training for realistic outputs

## Metrics

- **Style Similarity**: Cosine similarity between style features
- **Content Preservation**: MFCC distance between original and transferred content
- **Perceptual Quality**: Spectral distance and harmonicity measures
- **Temporal Consistency**: Frame-level style consistency

## Limitations

- Requires paired content-style datasets for training
- Quality depends on style-content compatibility
- Computational requirements for real-time processing
- May not preserve all musical nuances perfectly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{music_style_transfer,
  title={Music Style Transfer using Neural Networks},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Music-Style-Transfer}
}
```

## Acknowledgments

- Librosa for audio processing
- PyTorch for deep learning framework
- The open-source audio processing community
# Music-Style-Transfer
