"""
Neural embedding model for R2‑D2 phrase recognition.
===================================================

This module defines a simple Siamese network intended for learning
embeddings of R2‑D2 audio signals.  The goal of such a model is to
map similar phrases to nearby points in a vector space while pushing
dissimilar phrases apart.  When integrated into the recogniser the
embedding replaces or augments the DTW+MFCC distance metric.

The implementation uses PyTorch if available.  If PyTorch is not
installed the functions gracefully fall back to no‑ops so that the
rest of the system continues to function without the neural model.

The ``SiameseNet`` architecture is deliberately lightweight: a
sequence of one‑dimensional convolutions followed by global average
pooling.  It operates on sequences of MFCC features rather than raw
audio; this reduces the dimensionality of the input and allows the
network to focus on timbral structure rather than absolute time.

Example usage:

    from embedding_model import load_model, compute_embedding
    model = load_model('embedding.pth')
    if model is not None:
        embed = compute_embedding(audio_array, sample_rate, model)
        # compare embed to stored embeddings

"""

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from recognizer import Recognizer  # reuse MFCC computation


if TORCH_AVAILABLE:
    class SiameseNet(nn.Module):
        """A very small convolutional neural network for audio embeddings."""

        def __init__(self, input_dim: int = 13, embedding_dim: int = 32) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(128, embedding_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, channels, time)
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = torch.relu(self.bn3(self.conv3(x)))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    def load_model(path: str) -> Optional["SiameseNet"]:
        """Load a trained embedding model from disk.

        Args:
            path: Path to a ``.pth`` file containing model weights.

        Returns:
            A ``SiameseNet`` instance in evaluation mode, or ``None`` if
            PyTorch is unavailable or loading fails.
        """
        if not TORCH_AVAILABLE:
            return None
        model = SiameseNet()
        try:
            state = torch.load(path, map_location='cpu')
            model.load_state_dict(state)
            model.eval()
            return model
        except Exception:
            return None

    def compute_embedding(audio: np.ndarray, sample_rate: int, model: "SiameseNet") -> Optional[np.ndarray]:
        """Compute an embedding for a single audio clip using the given model.

        Args:
            audio: 1‑D numpy array of the signal.
            sample_rate: Sampling rate of the signal.
            model: A loaded ``SiameseNet`` instance.

        Returns:
            A 1‑D numpy array of the embedding, or ``None`` if MFCC
            extraction fails.
        """
        if not TORCH_AVAILABLE or model is None:
            return None
        # Use the recogniser's MFCC computation without delta and delta‑delta
        recogniser = Recognizer(sample_rate=sample_rate)
        feats = recogniser._compute_features(audio)
        if feats.shape[0] == 0:
            return None
        # Only use the first 13 MFCC coefficients
        mfcc = feats[:, :13].T  # shape (channels, time)
        with torch.no_grad():
            x = torch.from_numpy(mfcc).unsqueeze(0).float()
            emb = model(x).squeeze(0).numpy()
        return emb
else:
    # Fallback stubs if PyTorch is unavailable
    SiameseNet = None  # type: ignore
    def load_model(path: str) -> None:
        return None
    def compute_embedding(audio: np.ndarray, sample_rate: int, model: None) -> None:
        return None