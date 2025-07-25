"""
Synthesiser for R2‑D2 style audio.
=================================

This module implements a simple sound synthesis engine to transform
arbitrary English text into a sequence of beeps reminiscent of the
Star Wars character R2‑D2.  It consumes a ``lexicon.json`` file
containing three mappings:

* ``phrases`` – full sentences or multi‑word utterances that map to
  one or more sound segments.
* ``common_words`` – high frequency individual words such as "yes",
  "no" or "help" that map to short sequences of segments.
* ``alphabet`` – a fallback mapping from the Roman alphabet (A–Z)
  to a single segment each.

Each segment in the lexicon describes the following parameters:

  - ``freq``: Base frequency in Hz.
  - ``dur``: Duration in milliseconds.
  - ``env``: Name of the amplitude envelope; one of ``linear``,
    ``attack-decay`` or ``s-curve``.
  - ``filter``: A dictionary specifying an IIR filter with keys
    ``type`` (``lowpass``, ``highpass`` or ``bandpass``), ``cutoff``
    (Hz) and ``Q`` (quality factor).  Bandpass filters use the
    cutoff as the centre frequency and derive the bandwidth from
    ``Q`` (bandwidth = centre / Q).
  - ``vibrato``: Frequency modulation parameters with ``rate`` (Hz)
    and ``depth`` (Hz).  If ``rate`` is zero then no vibrato is
    applied.
  - ``noise_mix``: Weight of white noise mixed into the signal
    (0.0–1.0).
  - ``random_shift``: Random perturbations applied on each call.  It
    is a dictionary with ``freq`` (maximum frequency deviation in
    Hz) and ``dur`` (maximum duration deviation in milliseconds).

The synthesiser does not depend on external audio libraries like
``pydub``.  Instead it relies solely on ``numpy`` and
``scipy.signal`` to construct waveforms.  The ``generate_r2``
function returns a tuple ``(audio, sample_rate)`` where ``audio`` is
a floating‑point numpy array in ``[-1, 1]`` and ``sample_rate`` is
the sampling frequency used (default is 22050 Hz).

Example:

    from synth import generate_r2
    audio, sr = generate_r2("hello there")
    from scipy.io import wavfile
    wavfile.write('hello_there.wav', sr, (audio * 32767).astype('int16'))

"""

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal


class R2Synth:
    """Encapsulates lexicon and synthesis logic for R2‑D2 sounds."""

    def __init__(self, lexicon_path: str = None, sample_rate: int = 22050, *, cross_fade_ms: float = 0.0, waveform_mix: Tuple[float, float] = (0.7, 0.3), version: str = None) -> None:
        # Resolve lexicon path; fall back to same directory as this module.
        # Resolve lexicon path.  If a version is supplied, append
        # ``_v{version}.json`` to the base filename.  Otherwise use
        # the provided path or default to ``lexicon.json``.
        base_dir = os.path.dirname(__file__)
        if lexicon_path is None:
            base_name = 'lexicon'
            if version:
                lexicon_path = os.path.join(base_dir, f'{base_name}_v{version}.json')
            else:
                lexicon_path = os.path.join(base_dir, 'lexicon.json')
        else:
            # If lexicon_path is provided and version is given, append
            # version suffix to the filename if not already present.
            if version and not lexicon_path.endswith(f'_v{version}.json'):
                root, ext = os.path.splitext(lexicon_path)
                lexicon_path = f'{root}_v{version}{ext}'
        with open(lexicon_path, 'r') as fh:
            lex = json.load(fh)
        # Store lexicon path and modification time for hot reload
        self._lexicon_path = lexicon_path
        self._lexicon_mtime = os.path.getmtime(lexicon_path)
        # Lowercase keys for phrases and common words; uppercase for alphabet.
        self.phrases: Dict[str, List[Dict]] = {k.lower(): v for k, v in lex.get('phrases', {}).items()}
        self.common: Dict[str, List[Dict]] = {k.lower(): v for k, v in lex.get('common_words', {}).items()}
        self.alphabet: Dict[str, Dict] = {k.upper(): v for k, v in lex.get('alphabet', {}).items()}
        self.sample_rate: int = sample_rate
        # Cross‑fade duration in milliseconds.  A small non‑zero value
        # produces smoother transitions between segments.  Zero
        # disables cross‑fading.
        self.cross_fade_ms: float = max(0.0, float(cross_fade_ms))
        # Proportion of sine and square in the base waveform.  Must
        # sum to 1.  These weights can be adjusted at instantiation.
        sw, sq = waveform_mix
        total = sw + sq if (sw + sq) != 0 else 1.0
        self.waveform_mix: Tuple[float, float] = (sw / total, sq / total)
        # Cache for immutable segments (ones with no random shift).  Maps
        # a canonical string representation of the segment to an array.
        self._segment_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Lexicon hot reload
    def _maybe_reload(self) -> None:
        """Check whether the lexicon file has changed and reload if necessary."""
        try:
            mtime = os.path.getmtime(self._lexicon_path)
        except OSError:
            return
        if mtime != self._lexicon_mtime:
            try:
                with open(self._lexicon_path, 'r') as fh:
                    lex = json.load(fh)
                self.phrases = {k.lower(): v for k, v in lex.get('phrases', {}).items()}
                self.common = {k.lower(): v for k, v in lex.get('common_words', {}).items()}
                self.alphabet = {k.upper(): v for k, v in lex.get('alphabet', {}).items()}
                self._lexicon_mtime = mtime
                # Invalidate caches because the mapping may have changed
                self._segment_cache.clear()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Envelope functions
    def _envelope(self, env_type: str, length: int) -> np.ndarray:
        """Return an amplitude envelope array.

        Supported envelopes:
        ``linear`` – decays linearly from 1 to 0.
        ``attack-decay`` – rises from 0 to 1, then falls back to 0.
        ``s-curve`` – smooth sinusoidal attack and decay.
        Unknown names default to the linear envelope.

        Args:
            env_type: Name of the envelope.
            length: Number of samples.

        Returns:
            A numpy array of shape (length,) with values in [0,1].
        """
        if length <= 0:
            return np.zeros(0, dtype=np.float32)
        if env_type == 'attack-decay':
            half = length // 2
            attack = np.linspace(0.0, 1.0, half, endpoint=False)
            decay = np.linspace(1.0, 0.0, length - half)
            return np.concatenate((attack, decay)).astype(np.float32)
        elif env_type == 's-curve':
            # Squared sine gives a bell‑shaped envelope peaking at 1.
            return (np.sin(np.linspace(0.0, np.pi, length)) ** 2).astype(np.float32)
        else:
            return np.linspace(1.0, 0.0, length).astype(np.float32)

    # ------------------------------------------------------------------
    # Filter design helper
    def _design_filter(self, filt: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Design an IIR filter according to the provided specification.

        Args:
            filt: Dictionary with keys ``type``, ``cutoff`` and ``Q``.

        Returns:
            Filter numerator (b) and denominator (a) arrays for use with
            ``signal.filtfilt`` or ``signal.lfilter``.
        """
        ftype = str(filt.get('type', 'lowpass')).lower()
        cutoff = float(filt.get('cutoff', 1000.0))
        Q = float(filt.get('Q', 1.0))
        nyq = self.sample_rate / 2.0
        # Normalise frequencies
        if ftype == 'bandpass':
            bw = max(cutoff / max(Q, 1e-6), 1.0)
            low = max(cutoff - bw / 2.0, 20.0)
            high = min(cutoff + bw / 2.0, nyq - 20.0)
            if high <= low:
                # Fallback to a lowpass filter if the computed band is invalid.
                norm_cut = min(max(cutoff / nyq, 1e-4), 0.99)
                b, a = signal.butter(2, norm_cut, btype='low')
            else:
                Wn = [low / nyq, high / nyq]
                b, a = signal.butter(2, Wn, btype='band')
        elif ftype == 'highpass':
            norm_cut = min(max(cutoff / nyq, 1e-4), 0.99)
            b, a = signal.butter(2, norm_cut, btype='high')
        else:
            norm_cut = min(max(cutoff / nyq, 1e-4), 0.99)
            b, a = signal.butter(2, norm_cut, btype='low')
        return b, a

    # ------------------------------------------------------------------
    # Segment synthesis
    def _generate_segment(self, params: Dict) -> np.ndarray:
        """Synthesize a single segment according to the lexicon parameters.

        This method constructs a mixture of sine and square waves,
        applies vibrato, mixes white noise, applies an amplitude
        envelope and finally filters the result.  Randomised pitch
        and duration perturbations are applied on each call.

        Args:
            params: Segment specification from the lexicon.

        Returns:
            A numpy array of floating samples in the range [-1,1].
        """
        # Extract base parameters
        base_freq = float(params.get('freq', 440.0))
        base_dur_ms = float(params.get('dur', 200.0))
        env_type = str(params.get('env', 'linear'))
        filt_spec = params.get('filter', {})
        vibrato_spec = params.get('vibrato', {})
        noise_mix = float(params.get('noise_mix', 0.0))
        rand_spec = params.get('random_shift', {})
        # Random perturbations
        freq_shift = float(rand_spec.get('freq', 0.0))
        dur_shift = float(rand_spec.get('dur', 0.0))
        # Check cache eligibility: only cache when no random shift is specified.
        if freq_shift == 0.0 and dur_shift == 0.0:
            # Build a canonical representation of the immutable fields.
            import json as _json
            cache_key = _json.dumps({
                'freq': base_freq,
                'dur': base_dur_ms,
                'env': env_type,
                'filter': filt_spec,
                'vibrato': vibrato_spec,
                'noise_mix': noise_mix
            }, sort_keys=True, default=str)
            cached = self._segment_cache.get(cache_key)
            if cached is not None:
                return cached.copy()
        # Apply random shifts uniformly within ±specified ranges
        if freq_shift > 0.0:
            shift = random.uniform(-freq_shift, freq_shift)
            freq = max(20.0, base_freq + shift)
        else:
            freq = base_freq
        if dur_shift > 0.0:
            dur_ms = max(10.0, base_dur_ms + random.uniform(-dur_shift, dur_shift))
        else:
            dur_ms = base_dur_ms
        duration_sec = dur_ms / 1000.0
        n_samples = max(1, int(self.sample_rate * duration_sec))
        t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)
        # Vibrato parameters
        vibrato_rate = float(vibrato_spec.get('rate', 0.0))
        vibrato_depth = float(vibrato_spec.get('depth', 0.0))
        # Emotion mapping: adjust frequency and vibrato_rate according to
        # optional "emotion" tag in the segment specification.  This
        # allows prosodic variation without duplicating entire entries
        emotion = params.get('emotion')
        if isinstance(emotion, str):
            e = emotion.lower().strip()
            if e == 'urgent':
                freq *= 1.2
                vibrato_rate *= 1.5 if vibrato_rate > 0 else 1.5
            elif e == 'calm':
                freq *= 0.8
                vibrato_rate *= 0.5 if vibrato_rate > 0 else 0.5
            elif e == 'playful':
                freq *= 1.0 + random.uniform(-0.1, 0.1)
                vibrato_rate *= 1.0 + random.uniform(-0.3, 0.3) if vibrato_rate > 0 else 1.0
        # Compute phase with vibrato.  See module docstring for derivation.
        if vibrato_rate > 0.0 and vibrato_depth != 0.0:
            phi = 2.0 * np.pi * freq * t + (vibrato_depth / vibrato_rate) * (1.0 - np.cos(2.0 * np.pi * vibrato_rate * t))
        else:
            phi = 2.0 * np.pi * freq * t
        # Generate base waveform.  Use weights defined at instantiation.
        sine = np.sin(phi)
        square = np.sign(np.sin(phi))
        base = self.waveform_mix[0] * sine + self.waveform_mix[1] * square
        # Envelope
        env = self._envelope(env_type, n_samples)
        sig = base * env
        # Noise mixing
        if noise_mix > 0.0:
            noise = np.random.normal(scale=1.0, size=n_samples)
            if np.max(np.abs(noise)) > 0:
                noise = noise / np.max(np.abs(noise))
            sig = (1.0 - noise_mix) * sig + noise_mix * noise
        # Filtering
        b, a = self._design_filter(filt_spec)
        try:
            sig_filt = signal.filtfilt(b, a, sig)
        except ValueError:
            # filtfilt requires a minimum length; fall back to lfilter on tiny signals
            sig_filt = signal.lfilter(b, a, sig)
        # Normalise amplitude
        max_val = np.max(np.abs(sig_filt)) if np.max(np.abs(sig_filt)) > 0 else 1.0
        result = (sig_filt / max_val).astype(np.float32)
        # Store immutable segments in cache
        if freq_shift == 0.0 and dur_shift == 0.0:
            self._segment_cache[cache_key] = result.copy()
        return result

    # ------------------------------------------------------------------
    # Phrase synthesis helpers
    def _lookup_sequence(self, phrase: str) -> List[Dict]:
        """Resolve a phrase into a sequence of segment specifications.

        The lookup order is:

        * If the entire phrase appears in ``common_words`` then its
          definition is returned.
        * Otherwise if the phrase appears in ``phrases`` then that
          definition is used.
        * Otherwise the phrase is broken into words; words found in
          ``common_words`` are looked up individually and the segments
          concatenated.
        * If no matching words are found then the phrase is spelt out
          letter by letter using the ``alphabet`` mapping.

        Args:
            phrase: Input text (case insensitive).

        Returns:
            A list of segment dictionaries.
        """
        key = phrase.strip().lower()
        if key in self.common:
            return self.common[key]
        if key in self.phrases:
            return self.phrases[key]
        # Word‑level fallback: break the phrase into tokens and stitch
        # together any common‑word segments that exist.
        parts: List[str] = [p for p in key.split() if p]
        segments: List[Dict] = []
        found_any = False
        for word in parts:
            if word in self.common:
                segments.extend(self.common[word])
                found_any = True
            elif word in self.phrases:
                segments.extend(self.phrases[word])
                found_any = True
        if found_any:
            return segments
        # Final fallback: spell out character by character.
        result = []
        for char in phrase:
            letter = char.upper()
            if letter in self.alphabet:
                result.append(self.alphabet[letter])
        return result

    def generate_r2(self, phrase: str, *, apply_reverb: bool = False) -> Tuple[np.ndarray, int]:
        """Generate R2‑D2 audio for an arbitrary phrase.

        The lexicon lookup described in ``_lookup_sequence`` is used
        internally.  All segments are synthesised and concatenated in
        order.  The returned waveform is normalised so the largest
        sample lies within ±1.

        Args:
            phrase: Input text.

        Returns:
            A tuple ``(audio, sample_rate)`` where ``audio`` is a
            float32 numpy array and ``sample_rate`` is the sampling
            frequency used.
        """
        # Reload the lexicon if the underlying file has changed
        self._maybe_reload()
        segments = self._lookup_sequence(phrase)
        if not segments:
            # If nothing was found, return a very short silence to avoid
            # crashing the rest of the pipeline.
            return np.zeros(1, dtype=np.float32), self.sample_rate
        waves: List[np.ndarray] = []
        for seg in segments:
            waves.append(self._generate_segment(seg))
        # Concatenate with optional cross‑fading
        if not waves:
            return np.zeros(1, dtype=np.float32), self.sample_rate
        output = waves[0].copy()
        fade_samples = int(self.cross_fade_ms * self.sample_rate / 1000.0)
        for w in waves[1:]:
            if fade_samples > 0 and len(output) >= fade_samples and len(w) >= fade_samples:
                prev_tail = output[-fade_samples:]
                next_head = w[:fade_samples]
                # Linear cross‑fade weights
                weights_prev = np.linspace(1.0, 0.0, fade_samples, endpoint=False)
                weights_next = 1.0 - weights_prev
                cross = prev_tail * weights_prev + next_head * weights_next
                output = np.concatenate((output[:-fade_samples], cross, w[fade_samples:]))
            else:
                output = np.concatenate((output, w))
        # Normalise overall audio
        max_val = np.max(np.abs(output)) if np.max(np.abs(output)) > 0 else 1.0
        audio = (output / max_val).astype(np.float32)
        # Optional post‑processing: simple reverb
        if apply_reverb:
            audio = self._apply_reverb(audio)
        return audio, self.sample_rate

    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Apply a very simple reverb (echo) effect.

        The implementation convolves the input signal with an
        exponentially decaying impulse response.  The decay constant
        controls the length of the reverb tail.  This is a cheap
        approximation and can be replaced by more sophisticated
        convolution with real room responses.

        Args:
            audio: The dry signal.

        Returns:
            A new numpy array containing the reverberated signal.
        """
        # Length of reverb impulse in seconds
        impulse_length = 0.2  # 200 ms
        decay = 0.05  # time constant of exponential decay
        n_imp = int(self.sample_rate * impulse_length)
        # Build impulse response: exp(-t/decay)
        t = np.linspace(0, impulse_length, n_imp, endpoint=False)
        impulse = np.exp(-t / decay)
        impulse = impulse / np.max(np.abs(impulse))
        # Convolve using FFT for efficiency
        conv = signal.fftconvolve(audio, impulse)[:len(audio)]
        # Mix dry and wet signals
        wet_mix = 0.3
        out = (1.0 - wet_mix) * audio + wet_mix * conv
        # Normalise
        max_val = np.max(np.abs(out)) if np.max(np.abs(out)) > 0 else 1.0
        return (out / max_val).astype(np.float32)


def generate_r2(phrase: str) -> Tuple[np.ndarray, int]:
    """Convenience wrapper around :class:`R2Synth` for one‑off synthesis.

    A new ``R2Synth`` instance is created on every call.  This is
    sufficient for scripts or simple servers but for repeated calls
    (e.g. in a web service) it is more efficient to create a single
    ``R2Synth`` and reuse it.

    Args:
        phrase: The phrase to synthesise.

    Returns:
        A tuple ``(audio, sample_rate)``.
    """
    synth = R2Synth()
    return synth.generate_r2(phrase)