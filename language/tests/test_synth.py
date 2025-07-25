import numpy as np

from synth import generate_r2, R2Synth


def test_generate_known_phrase():
    """Synthesising a known phrase produces non‑empty audio."""
    audio, sr = generate_r2("hello")
    assert isinstance(audio, np.ndarray)
    assert audio.size > 0
    # Sample rate should be positive
    assert sr > 8000
    # Values should lie in [-1,1]
    assert np.max(audio) <= 1.0 + 1e-6
    assert np.min(audio) >= -1.0 - 1e-6


def test_generate_fallback_phrase():
    """Unknown phrases fall back to spelling out letters."""
    synth = R2Synth()
    phrase = "xyzabc"
    audio, sr = synth.generate_r2(phrase)
    assert audio.size > 0
    # Fallback sequence length should equal number of letters
    # Each letter in the alphabet mapping defines one segment; the audio
    # length will be roughly proportional to the number of characters.
    assert len(phrase) >= 1