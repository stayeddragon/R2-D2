"""
Celery task definitions for the R2‑D2 translator.
================================================

This module registers long‑running synthesis and recognition tasks
with Celery.  Offloading these jobs to background workers allows
your FastAPI server to remain responsive under heavy load.  Tasks
communicate with the rest of the application by serialising
arguments and return values as JSON; binary data such as audio
segments are encoded as base64 strings.

Two tasks are provided:

* ``synthesize_phrase`` – Generate the R2‑D2 audio for a phrase
  using the ``R2Synth``.  This task accepts a phrase, optional
  lexicon version and optional reverb flag.  It returns a base64
  encoded WAV file (16‑bit PCM).

* ``recognize_audio`` – Recognise the phrase encoded in a WAV
  payload.  The payload should be a base64 string of the raw PCM
  bytes.  The task decodes the audio, runs recognition and returns
  the best match, second best match and a confidence score.

If Celery or one of its dependencies is not installed, importing
this module will raise an ``ImportError``.  In that case the API
will fall back to synchronous execution (see ``server.py``).
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Optional, Tuple

from celery import Celery
from scipy.io import wavfile
import numpy as np

from synth import R2Synth
from recognizer import Recognizer


# Initialise Celery application.  The name ``tasks`` is conventional.
celery_app = Celery("tasks")
celery_app.config_from_object("celeryconfig")


@celery_app.task(name="tasks.synthesize_phrase")
def synthesize_phrase(phrase: str, version: Optional[str] = None, reverb: bool = False) -> str:
    """Generate R2‑D2 audio for a phrase and return a base64 WAV.

    Args:
        phrase: The text to synthesise.
        version: Optional lexicon version suffix.
        reverb: Whether to apply simple reverberation.

    Returns:
        A base64 encoded string of the 16‑bit PCM WAV data.
    """
    # Instantiate synthesiser lazily on worker.  Lexicon version may
    # refer to a file ``lexicon_v{version}.json``; if it does not
    # exist the synthesiser will raise an exception which Celery
    # propagates to the caller.
    synth = R2Synth(version=version)
    audio, sr = synth.generate_r2(phrase, apply_reverb=reverb)
    pcm = (audio * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@celery_app.task(name="tasks.recognize_audio")
def recognize_audio(wav_b64: str, version: Optional[str] = None, return_all: bool = False) -> dict:
    """Recognise a phrase from a base64 WAV payload.

    Args:
        wav_b64: Base64 encoded bytes of a WAV file containing 16‑bit
            PCM mono samples.
        version: Optional lexicon version.
        return_all: If ``True``, return the second best match and
            confidence score.

    Returns:
        A dictionary with keys ``text``; optionally ``second_best``
        and ``confidence`` if ``return_all`` is true.
    """
    # Decode WAV bytes
    wav_bytes = base64.b64decode(wav_b64)
    try:
        sr, samples = wavfile.read(io.BytesIO(wav_bytes))
    except Exception as e:
        raise ValueError(f"Could not read audio: {e}")
    # Convert to mono float32
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.dtype.kind in "iu":
        max_val = np.iinfo(samples.dtype).max
        audio = samples.astype(np.float32) / max_val
    elif samples.dtype.kind == "f":
        audio = samples.astype(np.float32)
    else:
        raise ValueError("Unsupported sample format")
    recogniser = Recognizer(lexicon_path=None if not version else os.path.join(os.path.dirname(__file__), f"lexicon_v{version}.json"), sample_rate=sr)
    if return_all:
        best, second, conf = recogniser.recognize(audio, sr, return_all=True)
        return {"text": best, "second_best": second, "confidence": float(conf)}
    else:
        text = recogniser.recognize(audio, sr, return_all=False)
        return {"text": text}
