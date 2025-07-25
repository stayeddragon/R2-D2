"""
Train a neural Siamese embedding model for R2‑D2 recognition.
===========================================================

This script provides a simple training loop for the ``SiameseNet``
defined in ``embedding_model.py``.  The goal is to learn an
embedding for each R2‑D2 phrase such that phrases map to distinct
regions of a low‑dimensional space.  Pairs of examples from the same
phrase (positives) should have embeddings that are close together,
while examples from different phrases (negatives) should be far
apart.  The network is trained using the contrastive loss.

Usage::

    python train_embedding.py --epochs 10 --lr 1e-3 --output embedding.pth

If PyTorch is not installed the script will print a message and
exit.  The training data are generated on the fly from the lexicon
loaded by ``Recognizer``; for each phrase we synthesise two
independent examples via ``R2Synth.generate_r2``.  Negative pairs
are sampled uniformly from distinct phrases.  You may adjust the
batch size, number of samples per phrase and margin to suit your
computing resources.

"""

from __future__ import annotations

import argparse
import itertools
import os
import random
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None

from embedding_model import SiameseNet
from recognizer import Recognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Siamese embedding model for R2‑D2 recognition.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="embedding.pth", help="Path to save the trained model")
    parser.add_argument("--samples_per_phrase", type=int, default=3, help="Number of training examples per phrase")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for contrastive loss")
    return parser.parse_args()


def collate_batch(features1: List[np.ndarray], features2: List[np.ndarray], labels: List[int], device: torch.device, max_len: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad or truncate feature sequences and convert to tensors.

    Each feature array has shape ``(n_frames, 26)`` (MFCC + delta).  We
    keep only the first 13 coefficients as input to the model.  All
    sequences are padded or truncated to ``max_len`` frames.  The
    resulting tensors have shape ``(batch, 13, max_len)``.

    Args:
        features1: List of feature matrices for the first item in each pair.
        features2: List of feature matrices for the second item in each pair.
        labels: List of integers (1 for positive, 0 for negative).
        device: Torch device.
        max_len: Maximum number of frames per sequence.

    Returns:
        Three tensors: ``x1``, ``x2`` and ``y``.
    """
    batch_size = len(features1)
    def pad(seq: np.ndarray) -> np.ndarray:
        # Use only the first 13 MFCCs
        mfcc = seq[:, :13]
        if mfcc.shape[0] >= max_len:
            return mfcc[:max_len, :]
        else:
            pad_width = max_len - mfcc.shape[0]
            return np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    x1 = np.stack([pad(f) for f in features1], axis=0)
    x2 = np.stack([pad(f) for f in features2], axis=0)
    y = np.array(labels, dtype=np.float32)
    # Convert to (batch, channels, time)
    x1_t = torch.tensor(x1.transpose(0, 2, 1), dtype=torch.float32, device=device)
    x2_t = torch.tensor(x2.transpose(0, 2, 1), dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    return x1_t, x2_t, y_t


def main() -> None:
    args = parse_args()
    if torch is None:
        print("PyTorch is not installed; cannot train embedding model.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate recogniser and synthesiser via recogniser
    recogniser = Recognizer(sample_rate=22050)
    phrases = list(recogniser.phrase_features.keys())
    if len(phrases) < 2:
        print("Need at least two phrases in the lexicon to train.")
        return
    # Create list of features per phrase by synthesising new examples
    def generate_features_for_phrase(phrase: str, n: int) -> List[np.ndarray]:
        feats = []
        for _ in range(n):
            audio, _ = recogniser.synth.generate_r2(phrase)
            f = recogniser._compute_features(audio)
            if f.shape[0] == 0:
                continue
            feats.append(f)
        return feats
    # Precompute features for all phrases
    phrase_to_feats = {p: generate_features_for_phrase(p, args.samples_per_phrase) for p in phrases}
    # Define model, loss and optimizer
    model = SiameseNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    margin = args.margin
    def contrastive_loss(emb1: torch.Tensor, emb2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Euclidean distance between embeddings
        diff = emb1 - emb2
        dist = torch.sqrt(torch.sum(diff * diff, dim=1) + 1e-9)
        # Contrastive loss: positive pairs (target=1) -> squared distance
        # negative pairs -> squared hinge on (margin - distance)
        loss_pos = target * dist * dist
        loss_neg = (1 - target) * torch.relu(margin - dist) * torch.relu(margin - dist)
        return torch.mean(loss_pos + loss_neg)
    # Training loop
    for epoch in range(args.epochs):
        # Shuffle phrases for each epoch
        random.shuffle(phrases)
        total_loss = 0.0
        count = 0
        # Generate pairs on the fly
        for i, phrase_i in enumerate(phrases):
            feats_i = phrase_to_feats[phrase_i]
            if not feats_i:
                continue
            # Pick random positive examples
            positives = list(itertools.combinations(feats_i, 2))
            # Pick negatives: pair each example from phrase_i with one example from another phrase
            negatives = []
            for phrase_j in phrases:
                if phrase_j == phrase_i:
                    continue
                feats_j = phrase_to_feats[phrase_j]
                if feats_j:
                    negatives.append((random.choice(feats_i), random.choice(feats_j)))
            # Combine and shuffle
            pairs = [(p[0], p[1], 1) for p in positives] + [(n[0], n[1], 0) for n in negatives]
            random.shuffle(pairs)
            # Mini‑batch processing (batch size = 8)
            batch_size = 8
            for b in range(0, len(pairs), batch_size):
                batch = pairs[b:b + batch_size]
                f1 = [p[0] for p in batch]
                f2 = [p[1] for p in batch]
                lbl = [p[2] for p in batch]
                x1_t, x2_t, y_t = collate_batch(f1, f2, lbl, device)
                model.train()
                optimizer.zero_grad()
                emb1 = model(x1_t)
                emb2 = model(x2_t)
                loss = contrastive_loss(emb1, emb2, y_t)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
                count += len(batch)
        if count > 0:
            avg_loss = total_loss / count
            print(f"Epoch {epoch+1}/{args.epochs} – avg loss: {avg_loss:.4f}")
    # Save model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
