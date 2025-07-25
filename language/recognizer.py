"""
Audio recognition for R2‑D2 beeps.
=================================

This module exposes a ``Recognizer`` class which can take an incoming
audio waveform and return the most probable phrase, using dynamic
time warping (DTW) over mel‑frequency cepstral coefficients (MFCCs).

Features are computed without relying on external libraries such as
``librosa``.  MFCC extraction and DTW are implemented from first
principles using ``numpy`` and ``scipy.signal``.  Delta features
(first‑order differences) are concatenated to the MFCCs to better
capture dynamics.

If the distance between the input utterance and all known phrases
exceeds a configurable threshold ``T``, the recognizer will fall back
to a spelled‑out representation.  In this case it splits the audio
into roughly equal segments, compares each segment against the
precomputed letter sounds (A–Z) and returns a string of the form
``"<spelled: ABC>"``.

Example:

    from recognizer import Recognizer
    recogniser = Recognizer()
    audio, sr = recogniser.synth.generate_r2("hello")
    phrase = recogniser.recognize(audio, sr)
    print(phrase)  # -> "hello"

"""

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal
from scipy.fftpack import dct as _dct

from synth import R2Synth


class Recognizer:
    """Recognises R2‑D2 audio by comparing against a lexicon using DTW."""

    def __init__(self, lexicon_path: str | None = None, sample_rate: int = 22050, threshold: float | None = None) -> None:
        """Initialise a recogniser.

        Args:
            lexicon_path: Optional path to a lexicon JSON file.  If
                ``None`` the default lexicon bundled with the package
                is used.  You can supply a versioned lexicon file
                (e.g. ``lexicon_v1.json``) to experiment with
                alternative mappings.
            sample_rate: Sampling frequency expected by the input
                audio.  The synthesiser uses the same rate for
                generated audio.
            threshold: Optional distance threshold.  If omitted the
                recogniser estimates a threshold from the inter‑phrase
                distances at startup.  A lower threshold makes the
                recogniser stricter.
        """
        # Instantiate synthesiser and load lexicon
        self.synth: R2Synth = R2Synth(lexicon_path=lexicon_path, sample_rate=sample_rate)
        self.sample_rate: int = self.synth.sample_rate
        # Precompute MFCC features for phrases and words.  Apply a small
        # noise augmentation to improve robustness: generate audio a
        # second time with white noise added and average the features.
        self.phrase_features: Dict[str, np.ndarray] = {}
        # Use both phrases and common words as keys
        for phrase_key in list(self.synth.phrases.keys()) + list(self.synth.common.keys()):
            if phrase_key in self.phrase_features:
                continue
            audio, _ = self.synth.generate_r2(phrase_key)
            feat_orig = self._compute_features(audio)
            # Augment with white noise; amplitude scaled relative to signal
            noise = np.random.normal(0.0, 0.02, size=audio.shape)
            audio_aug = np.clip(audio + noise, -1.0, 1.0)
            feat_aug = self._compute_features(audio_aug)
            # Average features if both are non‑empty
            if feat_orig.shape[0] > 0 and feat_aug.shape[0] > 0:
                # Pad sequences to equal length by truncating the longer one
                min_len = min(feat_orig.shape[0], feat_aug.shape[0])
                feat = 0.5 * (feat_orig[:min_len] + feat_aug[:min_len])
            else:
                feat = feat_orig
            self.phrase_features[phrase_key] = feat
        # Precompute a simple embedding (mean of MFCCs) for each phrase.
        # This acts as a crude neural fingerprint used to compute
        # confidence scores.  If the feature matrix is empty a zero
        # vector is stored.
        self.phrase_embeddings: Dict[str, np.ndarray] = {}
        for key, feat in self.phrase_features.items():
            if feat.shape[0] > 0:
                self.phrase_embeddings[key] = np.mean(feat[:, :13], axis=0)
            else:
                self.phrase_embeddings[key] = np.zeros(13, dtype=np.float32)
        # Precompute letter features for spelled‑out fallback
        self.letter_features: Dict[str, np.ndarray] = {}
        for letter, spec in self.synth.alphabet.items():
            audio = self.synth._generate_segment(spec)
            self.letter_features[letter] = self._compute_features(audio)
        # Determine threshold
        if threshold is None:
            # Estimate threshold from a sample of phrases.  Use up to 10
            # phrases to compute pairwise DTW distances and set the
            # threshold to half the mean distance.
            phrases = list(self.phrase_features.items())
            limit = min(10, len(phrases))
            total = 0.0
            count = 0
            for i in range(limit):
                for j in range(i + 1, limit):
                    d = self._dtw_distance(phrases[i][1], phrases[j][1])
                    total += d
                    count += 1
            self.threshold = (total / count) * 0.5 if count > 0 else 50.0
        else:
            self.threshold = float(threshold)

    # ------------------------------------------------------------------
    # MFCC computation
    def _mel_filterbank(self, n_fft: int, n_mels: int) -> np.ndarray:
        """Create a Mel filterbank matrix.

        Args:
            n_fft: The FFT size used for spectral analysis.
            n_mels: Number of mel bands.

        Returns:
            An array of shape (n_mels, n_fft//2+1) representing the
            filterbank.  Each row is a triangular filter.
        """
        # Define mel conversion
        def hz_to_mel(hz: float) -> float:
            return 2595.0 * math.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel: float) -> float:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        # Compute mel points
        mel_low = hz_to_mel(0.0)
        mel_high = hz_to_mel(self.sample_rate / 2.0)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / self.sample_rate).astype(int)
        # Create filterbank
        filterbank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(1, n_mels + 1):
            start = bin_points[i - 1]
            centre = bin_points[i]
            end = bin_points[i + 1]
            if centre == start:
                centre += 1
            if end == centre:
                end += 1
            # Rising slope
            for k in range(start, centre):
                if 0 <= k < filterbank.shape[1]:
                    filterbank[i - 1, k] = (k - start) / (centre - start)
            # Falling slope
            for k in range(centre, end):
                if 0 <= k < filterbank.shape[1]:
                    filterbank[i - 1, k] = (end - k) / (end - centre)
        return filterbank

    def _compute_features(self, audio: np.ndarray) -> np.ndarray:
        """Compute MFCC + delta features for a given audio clip.

        Args:
            audio: 1D numpy array of floating samples in [-1,1].

        Returns:
            2D numpy array with shape (n_frames, 26) containing
            13 MFCCs followed by 13 delta coefficients per frame.
        """
        # Pre‑emphasis to balance spectral tilt
        if len(audio) == 0:
            return np.zeros((0, 26), dtype=np.float32)
        pre_emph = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        # Frame parameters: 25 ms frames with 10 ms step
        frame_len = int(0.025 * self.sample_rate)
        frame_step = int(0.010 * self.sample_rate)
        signal_length = len(pre_emph)
        num_frames = 1 + max(0, (signal_length - frame_len) // frame_step)
        # Pad signal to ensure at least one frame
        pad_length = int((num_frames - 1) * frame_step + frame_len)
        if pad_length > signal_length:
            pre_emph = np.append(pre_emph, np.zeros(pad_length - signal_length))
        # Window function
        frames = np.lib.stride_tricks.as_strided(pre_emph, shape=(num_frames, frame_len), strides=(pre_emph.strides[0] * frame_step, pre_emph.strides[0])).copy()
        frames *= np.hamming(frame_len)
        # FFT and power spectrum
        n_fft = 512
        mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
        pow_frames = (1.0 / n_fft) * (mag_frames ** 2)
        # Mel filterbank
        filterbank = self._mel_filterbank(n_fft, n_mels=26)
        mel_energies = np.dot(pow_frames, filterbank.T)
        # Avoid log of zero
        mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
        log_mel = np.log(mel_energies)
        # Discrete cosine transform to get MFCCs
        # Use SciPy's DCT implementation since numpy.fft does not provide dct
        mfcc = _dct(log_mel, type=2, norm='ortho', axis=1)[:, :13]
        # Delta features (first derivative).  Use simple ±1 difference with replication at edges
        delta = np.zeros_like(mfcc)
        for i in range(mfcc.shape[0]):
            if i == 0:
                delta[i] = mfcc[min(1, mfcc.shape[0]-1)] - mfcc[0]
            elif i == mfcc.shape[0] - 1:
                delta[i] = mfcc[i] - mfcc[i - 1]
            else:
                delta[i] = (mfcc[i + 1] - mfcc[i - 1]) / 2.0
        # Delta‑delta (acceleration) features
        dd = np.zeros_like(mfcc)
        for i in range(mfcc.shape[0]):
            if i == 0:
                dd[i] = delta[min(1, delta.shape[0]-1)] - delta[0]
            elif i == delta.shape[0] - 1:
                dd[i] = delta[i] - delta[i - 1]
            else:
                dd[i] = (delta[i + 1] - delta[i - 1]) / 2.0
        # Concatenate MFCC, delta and delta‑delta
        feat = np.concatenate((mfcc, delta, dd), axis=1)
        # Mean‑variance normalisation across frames
        mean = np.mean(feat, axis=0)
        std = np.std(feat, axis=0)
        std[std == 0] = 1.0
        norm = (feat - mean) / std
        return norm.astype(np.float32)

    # ------------------------------------------------------------------
    # Dynamic Time Warping
    def _dtw_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute DTW distance between two feature sequences.

        Args:
            x: Feature matrix of shape (n, d).
            y: Feature matrix of shape (m, d).

        Returns:
            Normalised DTW cost.  Lower values indicate greater
            similarity.
        """
        n, d = x.shape
        m, _ = y.shape
        # Initialize a matrix with infinities
        dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        dtw[0, 0] = 0.0
        # Populate the matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(x[i - 1] - y[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        # Normalise by path length (sum of lengths)
        distance = dtw[n, m] / (n + m)
        return float(distance)

    # ------------------------------------------------------------------
    # Recognition
    def recognize(self, audio: np.ndarray, sample_rate: int, *, return_all: bool = False) -> str | Tuple[str, str, float]:
        """Recognise the most probable phrase represented by the input audio.

        Args:
            audio: Floating point numpy array of the raw signal.
            sample_rate: The sampling rate of the input signal.  If
                different to the synthesiser’s sample rate the audio
                will be resampled.

        Returns:
            If ``return_all`` is ``False`` the recognised phrase
            (string) if the distance is below ``threshold``;
            otherwise a spelled‑out string starting with
            ``"<spelled: "``.  If ``return_all`` is ``True`` a tuple
            ``(best, second_best, confidence)`` is returned.  The
            confidence is a heuristic score in the range (0,1] where
            higher values indicate a better match.
        """
        # Resample if needed
        if sample_rate != self.sample_rate:
            # Use Fourier method to resample
            num = int(len(audio) * float(self.sample_rate) / sample_rate)
            audio = signal.resample(audio, num)
        # Extract features from input
        features = self._compute_features(audio)
        if features.size == 0:
            return ""
        # Compare against each precomputed phrase using both DTW and a
        # simple embedding distance.  We compute a combined score
        # weighted towards DTW.  Also store scores for top‑n.
        # Compute mean MFCC embedding of the input
        if features.shape[0] > 0:
            mean_input = np.mean(features[:, :13], axis=0)
        else:
            mean_input = np.zeros(13, dtype=np.float32)
        scores: Dict[str, float] = {}
        for phrase, ref_feat in self.phrase_features.items():
            dtw_score = self._dtw_distance(features, ref_feat)
            embed_ref = self.phrase_embeddings.get(phrase)
            embed_score = np.linalg.norm(mean_input - embed_ref) if embed_ref is not None else 0.0
            combined = dtw_score * 0.7 + embed_score * 0.3
            scores[phrase] = combined
        # Sort phrases by combined score
        sorted_phrases = sorted(scores.items(), key=lambda item: item[1])
        best_phrase, best_score = sorted_phrases[0]
        if len(sorted_phrases) > 1:
            second_best_phrase, second_score = sorted_phrases[1]
        else:
            second_best_phrase, second_score = '', float('inf')
        confidence = 1.0 / (1.0 + best_score) if best_score != float('inf') else 0.0
        if not return_all:
            if best_score <= self.threshold:
                return best_phrase
            spelled = self._spell_out(audio)
            return f"<spelled: {spelled}>"
        return best_phrase, second_best_phrase, float(confidence)

    def _spell_out(self, audio: np.ndarray) -> str:
        """Spell out an utterance by matching segments to letters.

        This naive implementation slices the input into evenly sized
        segments based on the average duration of alphabet entries.
        Each segment is matched to the letter with the lowest DTW
        distance.

        Args:
            audio: Input audio (already resampled to the synthesiser
                rate).

        Returns:
            A string of uppercase letters representing the spelled
            interpretation of the input.
        """
        # Use simple energy‐based segmentation to find probable letter
        # boundaries.  Compute an amplitude envelope and detect
        # sufficiently long silent gaps.
        if len(audio) == 0:
            return ''
        abs_audio = np.abs(audio)
        max_amp = np.max(abs_audio)
        if max_amp == 0:
            return ''
        # Silence threshold as a fraction of the maximum amplitude
        thresh = 0.1 * max_amp
        sil = abs_audio < thresh
        # Minimum gap length of 30 ms for a letter boundary
        min_gap = int(0.03 * self.sample_rate)
        # Identify silent runs
        cuts: List[int] = []
        in_silence = False
        start = 0
        for i, flag in enumerate(sil):
            if flag and not in_silence:
                # start of a silent region
                in_silence = True
                start = i
            elif not flag and in_silence:
                # end of silent region
                length = i - start
                if length >= min_gap:
                    cuts.append(start + length // 2)
                in_silence = False
        # If we ended in silence, evaluate the trailing region
        if in_silence:
            length = len(audio) - start
            if length >= min_gap:
                cuts.append(start + length // 2)
        # Always include end of audio as a cut
        cuts = [c for c in cuts if 0 < c < len(audio)]
        cuts.append(len(audio))
        # If no silence detected, fall back to equal segmentation
        segments: List[np.ndarray] = []
        if not cuts or len(cuts) == 1:
            # Determine approximate letter count from alphabet durations
            durations = [float(spec.get('dur', 200.0)) for spec in self.synth.alphabet.values()]
            avg_dur = np.mean(durations) / 1000.0 if durations else 0.2
            n_letters = max(1, int(round(len(audio) / (avg_dur * self.sample_rate))))
            seg_len = max(1, len(audio) // n_letters)
            for i in range(n_letters):
                start_idx = i * seg_len
                end_idx = len(audio) if i == n_letters - 1 else (i + 1) * seg_len
                segments.append(audio[start_idx:end_idx])
        else:
            # Segment by detected cuts
            prev = 0
            for cut in cuts:
                seg = audio[prev:cut]
                if len(seg) > 0:
                    segments.append(seg)
                prev = cut
        # Match each segment to the closest letter
        letters = []
        for seg_audio in segments:
            seg_feat = self._compute_features(seg_audio)
            if seg_feat.size == 0:
                letters.append('')
                continue
            best_letter = ''
            best_score = float('inf')
            for letter, ref_feat in self.letter_features.items():
                score = self._dtw_distance(seg_feat, ref_feat)
                if score < best_score:
                    best_score = score
                    best_letter = letter
            letters.append(best_letter)
        return ''.join(letters)


def recognise_from_bytes(data: bytes, lexicon_path: str = None) -> str:
    """Convenience function to recognise audio stored in raw bytes.

    The input is assumed to be a WAV file.  SciPy's wavfile reader is
    used internally.  If the lexicon path is given it is passed to
    ``Recognizer``; otherwise the default lexicon is used.

    Args:
        data: Byte string containing a WAV file.
        lexicon_path: Optional path to ``lexicon.json``.

    Returns:
        Recognised phrase or spelled‑out string.
    """
    from scipy.io import wavfile
    import io
    sr, samples = wavfile.read(io.BytesIO(data))
    # Convert integer types to float in [-1,1]
    if samples.dtype.kind in 'iu':
        max_val = np.iinfo(samples.dtype).max
        samples = samples.astype(np.float32) / max_val
    elif samples.dtype.kind == 'f':
        samples = samples.astype(np.float32)
    else:
        raise TypeError('Unsupported audio sample type')
    recogniser = Recognizer(lexicon_path=lexicon_path)
    return recogniser.recognize(samples, sr)