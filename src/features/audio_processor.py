"""Audio processing utilities and feature extraction."""

import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from typing import Tuple, Optional, Union, List
from pathlib import Path


class AudioProcessor:
    """Audio processing and feature extraction utilities."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 13,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        preemphasis: float = 0.97
    ):
        """Initialize audio processor with specified parameters."""
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.preemphasis = preemphasis
        
        # Initialize mel filter bank
        self.mel_filter = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=self.fmax
        )
    
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr
    
    def save_audio(self, audio: np.ndarray, file_path: Union[str, Path], sr: int) -> None:
        """Save audio data to file."""
        sf.write(file_path, audio, sr)
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply preprocessing to audio signal."""
        # Pre-emphasis
        if self.preemphasis > 0:
            audio = librosa.effects.preemphasis(audio, coef=self.preemphasis)
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        return mfcc
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features from audio."""
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return chroma
    
    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral centroid from audio."""
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return centroid
    
    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate from audio."""
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        return zcr
    
    def extract_rhythm_features(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract rhythm-related features (tempo and beat tracking)."""
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return tempo, beats
    
    def extract_all_features(self, audio: np.ndarray) -> dict:
        """Extract all audio features."""
        features = {
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'mfcc': self.extract_mfcc(audio),
            'chroma': self.extract_chroma(audio),
            'spectral_centroid': self.extract_spectral_centroid(audio),
            'zero_crossing_rate': self.extract_zero_crossing_rate(audio),
        }
        
        # Add rhythm features
        tempo, beats = self.extract_rhythm_features(audio)
        features['tempo'] = tempo
        features['beats'] = beats
        
        return features
    
    def pad_or_truncate(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate audio to target length."""
        if len(audio) > target_length:
            # Truncate
            audio = audio[:target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        
        return audio
    
    def augment_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply data augmentation to audio."""
        # Random pitch shifting
        if np.random.random() < 0.3:
            pitch_shift = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
        
        # Random time stretching
        if np.random.random() < 0.3:
            stretch_factor = np.random.uniform(0.8, 1.2)
            audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
        
        # Add noise
        if np.random.random() < 0.3:
            noise_level = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, len(audio))
            audio = audio + noise
        
        return audio


def audio_to_tensor(audio: np.ndarray) -> torch.Tensor:
    """Convert numpy audio array to PyTorch tensor."""
    return torch.from_numpy(audio).float()


def tensor_to_audio(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy audio array."""
    return tensor.detach().cpu().numpy()


def compute_spectral_distance(spectrogram1: np.ndarray, spectrogram2: np.ndarray) -> float:
    """Compute spectral distance between two spectrograms."""
    # Ensure same shape
    min_frames = min(spectrogram1.shape[1], spectrogram2.shape[1])
    spec1 = spectrogram1[:, :min_frames]
    spec2 = spectrogram2[:, :min_frames]
    
    # Compute mean squared error
    mse = np.mean((spec1 - spec2) ** 2)
    
    return float(mse)


def compute_chroma_distance(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """Compute chroma distance between two chroma features."""
    # Ensure same shape
    min_frames = min(chroma1.shape[1], chroma2.shape[1])
    chroma1 = chroma1[:, :min_frames]
    chroma2 = chroma2[:, :min_frames]
    
    # Compute cosine similarity
    dot_product = np.sum(chroma1 * chroma2)
    norm1 = np.linalg.norm(chroma1)
    norm2 = np.linalg.norm(chroma2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0
    
    cosine_sim = dot_product / (norm1 * norm2)
    return float(1.0 - cosine_sim)
