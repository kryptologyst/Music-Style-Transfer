"""Streamlit demo application for music style transfer."""

import streamlit as st
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import tempfile
import os
from pathlib import Path

from src.models import StyleTransferModel
from src.features.audio_processor import AudioProcessor
from src.metrics import StyleTransferMetrics
from src.utils.device import get_device
from src.utils.config import get_default_config


# Page configuration
st.set_page_config(
    page_title="Music Style Transfer Demo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Privacy disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Privacy Disclaimer</h4>
    <p><strong>This is a research and educational demonstration.</strong> This software is NOT intended for production use, biometric identification, or voice cloning applications. Any misuse of this technology for creating deepfakes, impersonation, or unauthorized voice synthesis is strictly prohibited and may violate laws and ethical guidelines.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎵 Music Style Transfer Demo</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["cnn", "transformer", "gan"],
    help="Choose the neural network architecture for style transfer"
)

# Device selection
device_option = st.sidebar.selectbox(
    "Select Device",
    ["auto", "cpu", "cuda", "mps"],
    help="Choose the computation device"
)

# Load model
@st.cache_resource
def load_model(model_type: str, device: str):
    """Load the style transfer model."""
    try:
        device_obj = get_device(device)
        model = StyleTransferModel(model_type=model_type)
        model = model.to(device_obj)
        return model, device_obj
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, device = load_model(model_type, device_option)

if model is None:
    st.error("Failed to load model. Please check the configuration.")
    st.stop()

# Initialize components
processor = AudioProcessor()
metrics_calculator = StyleTransferMetrics(processor)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Upload Audio Files")
    
    # Content audio upload
    st.subheader("Content Audio")
    content_file = st.file_uploader(
        "Upload content audio file",
        type=['wav', 'mp3', 'flac'],
        help="This is the audio whose content (melody, harmony) will be preserved"
    )
    
    # Style audio upload
    st.subheader("Style Audio")
    style_file = st.file_uploader(
        "Upload style audio file",
        type=['wav', 'mp3', 'flac'],
        help="This is the audio whose style (tempo, timbre, arrangement) will be transferred"
    )
    
    # Process button
    if st.button("🎯 Apply Style Transfer", type="primary"):
        if content_file is not None and style_file is not None:
            try:
                # Process files
                with st.spinner("Processing audio files..."):
                    # Load content audio
                    content_bytes = content_file.read()
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_file.write(content_bytes)
                        content_audio, content_sr = processor.load_audio(tmp_file.name)
                        os.unlink(tmp_file.name)
                    
                    # Load style audio
                    style_bytes = style_file.read()
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_file.write(style_bytes)
                        style_audio, style_sr = processor.load_audio(tmp_file.name)
                        os.unlink(tmp_file.name)
                    
                    # Preprocess audio
                    content_audio = processor.preprocess_audio(content_audio)
                    style_audio = processor.preprocess_audio(style_audio)
                    
                    # Extract features
                    content_features = processor.extract_mel_spectrogram(content_audio)
                    style_features = processor.extract_mel_spectrogram(style_audio)
                    
                    # Convert to tensors
                    content_tensor = torch.from_numpy(content_features).float().unsqueeze(0).unsqueeze(0)
                    style_tensor = torch.from_numpy(style_features).float().unsqueeze(0).unsqueeze(0)
                    
                    # Move to device
                    content_tensor = content_tensor.to(device)
                    style_tensor = style_tensor.to(device)
                    
                    # Apply style transfer
                    with torch.no_grad():
                        stylized_features = model(content_tensor, style_tensor)
                    
                    # Convert back to audio (simplified - in practice, you'd need a vocoder)
                    stylized_audio = stylized_features.squeeze().cpu().numpy()
                    
                    # Store results in session state
                    st.session_state['content_audio'] = content_audio
                    st.session_state['style_audio'] = style_audio
                    st.session_state['stylized_audio'] = stylized_audio
                    st.session_state['content_sr'] = content_sr
                    st.session_state['style_sr'] = style_sr
                    
                    st.success("Style transfer completed successfully!")
                    
            except Exception as e:
                st.error(f"Error during style transfer: {str(e)}")
        else:
            st.warning("Please upload both content and style audio files.")

with col2:
    st.header("📊 Results & Analysis")
    
    if 'stylized_audio' in st.session_state:
        # Display audio waveforms
        st.subheader("Audio Waveforms")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Content Audio", "Style Audio", "Stylized Audio"),
            vertical_spacing=0.1
        )
        
        # Content audio
        content_audio = st.session_state['content_audio']
        time_axis = np.linspace(0, len(content_audio) / st.session_state['content_sr'], len(content_audio))
        fig.add_trace(
            go.Scatter(x=time_axis, y=content_audio, name="Content", line=dict(color='blue')),
            row=1, col=1
        )
        
        # Style audio
        style_audio = st.session_state['style_audio']
        time_axis = np.linspace(0, len(style_audio) / st.session_state['style_sr'], len(style_audio))
        fig.add_trace(
            go.Scatter(x=time_axis, y=style_audio, name="Style", line=dict(color='green')),
            row=2, col=1
        )
        
        # Stylized audio
        stylized_audio = st.session_state['stylized_audio']
        time_axis = np.linspace(0, len(stylized_audio) / st.session_state['content_sr'], len(stylized_audio))
        fig.add_trace(
            go.Scatter(x=time_axis, y=stylized_audio, name="Stylized", line=dict(color='red')),
            row=3, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display spectrograms
        st.subheader("Mel Spectrograms")
        
        # Create spectrogram subplots
        fig_spec = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Content Spectrogram", "Style Spectrogram", "Stylized Spectrogram"),
            vertical_spacing=0.1
        )
        
        # Content spectrogram
        content_spec = processor.extract_mel_spectrogram(content_audio)
        fig_spec.add_trace(
            go.Heatmap(
                z=content_spec,
                colorscale='Viridis',
                name="Content Spec"
            ),
            row=1, col=1
        )
        
        # Style spectrogram
        style_spec = processor.extract_mel_spectrogram(style_audio)
        fig_spec.add_trace(
            go.Heatmap(
                z=style_spec,
                colorscale='Viridis',
                name="Style Spec"
            ),
            row=2, col=1
        )
        
        # Stylized spectrogram
        stylized_spec = processor.extract_mel_spectrogram(stylized_audio)
        fig_spec.add_trace(
            go.Heatmap(
                z=stylized_spec,
                colorscale='Viridis',
                name="Stylized Spec"
            ),
            row=3, col=1
        )
        
        fig_spec.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_spec, use_container_width=True)
        
        # Evaluation metrics
        st.subheader("📈 Evaluation Metrics")
        
        try:
            metrics = metrics_calculator.compute_all_metrics(
                stylized_audio, content_audio, style_audio
            )
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Style Similarity",
                    f"{metrics['style_similarity']:.3f}",
                    help="How well the style is captured"
                )
                st.metric(
                    "Content Preservation",
                    f"{metrics['content_preservation']:.3f}",
                    help="How well the content is preserved"
                )
            
            with col2:
                st.metric(
                    "Rhythm Similarity",
                    f"{metrics['rhythm_similarity']:.3f}",
                    help="Tempo and beat alignment"
                )
                st.metric(
                    "Harmonicity",
                    f"{metrics['harmonicity']:.3f}",
                    help="Harmonic content stability"
                )
            
            with col3:
                st.metric(
                    "Overall Quality",
                    f"{metrics['overall_quality']:.3f}",
                    help="Combined quality score"
                )
                st.metric(
                    "Spectral Distance",
                    f"{metrics['spectral_distance']:.3f}",
                    help="Lower is better"
                )
            
            # Quality assessment
            st.subheader("🎯 Quality Assessment")
            overall_score = metrics['overall_quality']
            
            if overall_score >= 0.8:
                st.success("**Excellent**: High-quality style transfer with good balance between style capture and content preservation.")
            elif overall_score >= 0.6:
                st.info("**Good**: Decent style transfer capabilities with room for improvement.")
            elif overall_score >= 0.4:
                st.warning("**Fair**: Basic style transfer functionality but needs significant improvements.")
            else:
                st.error("**Poor**: Requires substantial improvements to achieve acceptable quality.")
            
        except Exception as e:
            st.error(f"Error computing metrics: {str(e)}")
        
        # Download results
        st.subheader("💾 Download Results")
        
        # Create download button for stylized audio
        stylized_bytes = io.BytesIO()
        sf.write(stylized_bytes, stylized_audio, st.session_state['content_sr'], format='WAV')
        stylized_bytes.seek(0)
        
        st.download_button(
            label="Download Stylized Audio",
            data=stylized_bytes.getvalue(),
            file_name="stylized_audio.wav",
            mime="audio/wav"
        )

# Model information
st.sidebar.header("Model Information")
model_info = model.get_model_info()

st.sidebar.metric("Model Type", model_info['model_type'].upper())
st.sidebar.metric("Parameters", f"{model_info['total_parameters']:,}")
st.sidebar.metric("Model Size", f"{model_info['model_size_mb']:.1f} MB")
st.sidebar.metric("Device", str(device))

# Instructions
st.sidebar.header("📖 Instructions")
st.sidebar.markdown("""
1. **Upload Content Audio**: The audio whose melody/harmony you want to preserve
2. **Upload Style Audio**: The audio whose style you want to transfer
3. **Click Apply Style Transfer**: Process the audio files
4. **View Results**: Analyze waveforms, spectrograms, and metrics
5. **Download**: Save the stylized audio

**Note**: This is a research demo. Results may vary based on audio quality and compatibility.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Music Style Transfer Demo - Research & Educational Use Only<br>
    ⚠️ Not for production use or biometric applications
</div>
""", unsafe_allow_html=True)
