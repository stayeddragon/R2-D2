"""
R2-D2 Voice Synthesis Engine

Converts human speech (or text via TTS) into R2-D2-style beeps and whistles.
This mimics the original approach used by Ben Burtt - taking human vocalizations
and transforming them through synthesis.

Pipeline:
    Text -> TTS -> Speech Audio -> Feature Extraction -> R2D2 Synthesis -> Output

Features extracted from speech:
    - Pitch contour (F0) -> mapped to beep frequencies
    - Energy/loudness -> mapped to beep amplitude
    - Voiced/unvoiced segments -> beeps vs noise bursts
    - Rhythm/timing -> preserved in output
    - Phoneme-like segmentation -> individual beep units
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from dataclasses import dataclass
from typing import Optional, Tuple, List
import io
import tempfile
import subprocess
import os


# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE = 22050  # Standard sample rate for synthesis

# R2D2 frequency range (based on analysis of original sounds)
R2D2_FREQ_MIN = 200    # Hz - lowest beeps
R2D2_FREQ_MAX = 4000   # Hz - highest whistles
R2D2_FREQ_CENTER = 800 # Hz - neutral/resting frequency

# Human speech pitch range (for mapping)
HUMAN_PITCH_MIN = 75   # Hz - low male voice
HUMAN_PITCH_MAX = 400  # Hz - high female/child voice


@dataclass
class ProsodyFrame:
    """Represents extracted prosodic features for a single time frame."""
    time: float           # Time in seconds
    pitch: float          # F0 in Hz (0 if unvoiced)
    energy: float         # RMS energy (0-1 normalized)
    is_voiced: bool       # Whether this frame is voiced
    is_silence: bool      # Whether this is silence


@dataclass
class R2D2Segment:
    """A single synthesized R2D2 sound segment."""
    start_time: float
    duration: float
    segment_type: str     # 'beep', 'whistle', 'chirp', 'noise'
    frequencies: np.ndarray  # Frequency contour
    amplitudes: np.ndarray   # Amplitude envelope


# ============================================================================
# Prosody Extraction
# ============================================================================

class ProsodyExtractor:
    """
    Extracts prosodic features from speech audio.

    Uses autocorrelation-based pitch detection and frame-by-frame
    energy computation. No external dependencies required.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_duration_ms: float = 25.0,
        hop_duration_ms: float = 10.0,
        pitch_min: float = HUMAN_PITCH_MIN,
        pitch_max: float = HUMAN_PITCH_MAX,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.hop_size = int(sample_rate * hop_duration_ms / 1000)
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max

        # Precompute lag bounds for pitch detection
        self.min_lag = int(sample_rate / pitch_max)
        self.max_lag = int(sample_rate / pitch_min)

    def extract(self, audio: np.ndarray) -> List[ProsodyFrame]:
        """
        Extract prosodic features from audio.

        Args:
            audio: Mono audio signal, normalized to [-1, 1]

        Returns:
            List of ProsodyFrame objects, one per time frame
        """
        # Ensure float
        audio = audio.astype(np.float64)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Compute global energy for silence threshold
        global_energy = np.sqrt(np.mean(audio ** 2))
        silence_threshold = global_energy * 0.1

        frames = []
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1

        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            frame_audio = audio[start:end]

            time = (start + self.frame_size // 2) / self.sample_rate
            energy = np.sqrt(np.mean(frame_audio ** 2))

            # Check for silence
            is_silence = energy < silence_threshold

            if is_silence:
                frames.append(ProsodyFrame(
                    time=time,
                    pitch=0,
                    energy=0,
                    is_voiced=False,
                    is_silence=True
                ))
                continue

            # Pitch detection via autocorrelation
            pitch, is_voiced = self._detect_pitch(frame_audio)

            frames.append(ProsodyFrame(
                time=time,
                pitch=pitch,
                energy=min(1.0, energy / global_energy) if global_energy > 0 else 0,
                is_voiced=is_voiced,
                is_silence=False
            ))

        return frames

    def _detect_pitch(self, frame: np.ndarray) -> Tuple[float, bool]:
        """
        Detect pitch using autocorrelation method.

        Returns:
            (pitch_hz, is_voiced) tuple
        """
        # Apply window
        windowed = frame * np.hanning(len(frame))

        # Compute autocorrelation
        corr = np.correlate(windowed, windowed, mode='full')
        corr = corr[len(corr)//2:]  # Take positive lags only

        # Normalize
        if corr[0] > 0:
            corr = corr / corr[0]

        # Find peaks in valid lag range
        if self.max_lag >= len(corr):
            return 0, False

        search_region = corr[self.min_lag:self.max_lag]

        if len(search_region) == 0:
            return 0, False

        # Find the highest peak
        peak_idx = np.argmax(search_region)
        peak_value = search_region[peak_idx]

        # Voicing decision based on peak strength
        voicing_threshold = 0.3
        is_voiced = peak_value > voicing_threshold

        if not is_voiced:
            return 0, False

        # Convert lag to frequency
        lag = self.min_lag + peak_idx
        pitch = self.sample_rate / lag

        return pitch, True


# ============================================================================
# R2D2 Synthesizer
# ============================================================================

class R2D2Synthesizer:
    """
    Synthesizes R2-D2 style sounds from prosodic features.

    Maps human speech characteristics to R2D2-like beeps and whistles:
    - Pitch contour -> frequency contour (with expanded range)
    - Energy -> amplitude
    - Voiced segments -> tonal beeps/whistles
    - Unvoiced segments -> noise bursts or chirps
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        freq_min: float = R2D2_FREQ_MIN,
        freq_max: float = R2D2_FREQ_MAX,
        expressiveness: float = 1.5,  # How much to exaggerate pitch movements
        chirpiness: float = 0.3,      # Probability of adding frequency sweeps
        noise_blend: float = 0.1,     # Amount of noise to blend in
    ):
        self.sample_rate = sample_rate
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.expressiveness = expressiveness
        self.chirpiness = chirpiness
        self.noise_blend = noise_blend

    def synthesize(self, prosody_frames: List[ProsodyFrame]) -> np.ndarray:
        """
        Synthesize R2D2 audio from prosody frames.

        Args:
            prosody_frames: List of ProsodyFrame from ProsodyExtractor

        Returns:
            Synthesized audio as numpy array
        """
        if not prosody_frames:
            return np.array([], dtype=np.float32)

        # Convert prosody to R2D2 parameters
        segments = self._prosody_to_segments(prosody_frames)

        # Synthesize each segment
        total_duration = prosody_frames[-1].time + 0.025  # Add last frame duration
        total_samples = int(total_duration * self.sample_rate)
        audio = np.zeros(total_samples, dtype=np.float64)

        for segment in segments:
            segment_audio = self._synthesize_segment(segment)
            start_sample = int(segment.start_time * self.sample_rate)
            end_sample = start_sample + len(segment_audio)

            if end_sample > len(audio):
                end_sample = len(audio)
                segment_audio = segment_audio[:end_sample - start_sample]

            if start_sample < len(audio):
                audio[start_sample:end_sample] += segment_audio

        # Normalize and apply soft clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        audio = np.tanh(audio * 1.5) / np.tanh(1.5)  # Soft saturation

        return audio.astype(np.float32)

    def _prosody_to_segments(self, frames: List[ProsodyFrame]) -> List[R2D2Segment]:
        """Convert prosody frames into R2D2 sound segments."""
        segments = []

        # Group consecutive voiced/unvoiced frames
        current_group = []
        current_type = None

        for frame in frames:
            if frame.is_silence:
                frame_type = 'silence'
            elif frame.is_voiced:
                frame_type = 'voiced'
            else:
                frame_type = 'unvoiced'

            if frame_type != current_type and current_group:
                # Process the completed group
                seg = self._frames_to_segment(current_group, current_type)
                if seg is not None:
                    segments.append(seg)
                current_group = []

            current_type = frame_type
            current_group.append(frame)

        # Process final group
        if current_group:
            seg = self._frames_to_segment(current_group, current_type)
            if seg is not None:
                segments.append(seg)

        return segments

    def _frames_to_segment(
        self,
        frames: List[ProsodyFrame],
        segment_type: str
    ) -> Optional[R2D2Segment]:
        """Convert a group of frames into an R2D2 segment."""
        if segment_type == 'silence':
            return None

        start_time = frames[0].time
        duration = frames[-1].time - frames[0].time + 0.01  # Add frame duration

        if duration < 0.01:  # Skip very short segments
            return None

        num_samples = int(duration * self.sample_rate)

        if segment_type == 'voiced':
            # Map pitch contour to R2D2 frequencies
            pitches = np.array([f.pitch for f in frames])
            pitches = np.maximum(pitches, HUMAN_PITCH_MIN)  # Clamp

            # Map human pitch range to R2D2 range with expressiveness
            pitch_normalized = (pitches - HUMAN_PITCH_MIN) / (HUMAN_PITCH_MAX - HUMAN_PITCH_MIN)
            pitch_normalized = np.clip(pitch_normalized, 0, 1)

            # Apply expressiveness (exaggerate deviations from center)
            center = 0.5
            pitch_normalized = center + (pitch_normalized - center) * self.expressiveness
            pitch_normalized = np.clip(pitch_normalized, 0, 1)

            # Map to R2D2 frequency range
            frequencies = self.freq_min + pitch_normalized * (self.freq_max - self.freq_min)

            # Interpolate to sample rate
            frequencies = np.interp(
                np.linspace(0, 1, num_samples),
                np.linspace(0, 1, len(frequencies)),
                frequencies
            )

            # Add some random frequency wobble for character
            wobble = np.sin(np.linspace(0, 20 * np.pi, num_samples)) * 30
            frequencies = frequencies + wobble

            # Maybe add chirp (frequency sweep)
            if np.random.random() < self.chirpiness:
                chirp_amount = np.random.uniform(-200, 200)
                chirp = np.linspace(0, chirp_amount, num_samples)
                frequencies = frequencies + chirp

            # Amplitude envelope from energy
            energies = np.array([f.energy for f in frames])
            amplitudes = np.interp(
                np.linspace(0, 1, num_samples),
                np.linspace(0, 1, len(energies)),
                energies
            )

            # Smooth the envelope
            window_size = min(100, num_samples // 4)
            if window_size > 1:
                amplitudes = np.convolve(
                    amplitudes,
                    np.ones(window_size) / window_size,
                    mode='same'
                )

            return R2D2Segment(
                start_time=start_time,
                duration=duration,
                segment_type='beep',
                frequencies=frequencies,
                amplitudes=amplitudes
            )

        else:  # unvoiced
            # Create noise burst or chirpy sounds for unvoiced segments
            frequencies = np.linspace(
                np.random.uniform(500, 1500),
                np.random.uniform(500, 1500),
                num_samples
            )

            energies = np.array([f.energy for f in frames])
            amplitudes = np.interp(
                np.linspace(0, 1, num_samples),
                np.linspace(0, 1, len(energies)),
                energies
            ) * 0.5  # Reduce volume for noise

            return R2D2Segment(
                start_time=start_time,
                duration=duration,
                segment_type='noise',
                frequencies=frequencies,
                amplitudes=amplitudes
            )

    def _synthesize_segment(self, segment: R2D2Segment) -> np.ndarray:
        """Synthesize audio for a single segment."""
        num_samples = len(segment.frequencies)
        t = np.arange(num_samples) / self.sample_rate

        if segment.segment_type == 'beep':
            # Generate frequency-modulated tone
            # Phase is integral of frequency
            phase = 2 * np.pi * np.cumsum(segment.frequencies) / self.sample_rate

            # Mix of sine and slightly squared wave for character
            sine = np.sin(phase)
            squared = np.tanh(3 * np.sin(phase))  # Soft squaring
            tone = 0.7 * sine + 0.3 * squared

            # Add harmonics for richness
            harmonic2 = 0.15 * np.sin(2 * phase)
            harmonic3 = 0.08 * np.sin(3 * phase)
            tone = tone + harmonic2 + harmonic3

            # Add subtle noise
            noise = np.random.randn(num_samples) * self.noise_blend
            tone = tone + noise

            # Apply amplitude envelope with attack/release
            envelope = segment.amplitudes.copy()

            # Attack (fade in)
            attack_samples = min(int(0.005 * self.sample_rate), num_samples // 4)
            if attack_samples > 0:
                envelope[:attack_samples] *= np.linspace(0, 1, attack_samples)

            # Release (fade out)
            release_samples = min(int(0.01 * self.sample_rate), num_samples // 4)
            if release_samples > 0:
                envelope[-release_samples:] *= np.linspace(1, 0, release_samples)

            return tone * envelope

        else:  # noise
            # Filtered noise burst
            noise = np.random.randn(num_samples)

            # Bandpass filter the noise
            center_freq = np.mean(segment.frequencies)
            bandwidth = 500
            low = max(100, center_freq - bandwidth)
            high = min(self.sample_rate / 2 - 100, center_freq + bandwidth)

            sos = signal.butter(
                2,
                [low, high],
                btype='band',
                fs=self.sample_rate,
                output='sos'
            )
            filtered_noise = signal.sosfilt(sos, noise)

            # Apply envelope
            envelope = segment.amplitudes.copy()
            attack_samples = min(int(0.003 * self.sample_rate), num_samples // 4)
            if attack_samples > 0:
                envelope[:attack_samples] *= np.linspace(0, 1, attack_samples)
            release_samples = min(int(0.005 * self.sample_rate), num_samples // 4)
            if release_samples > 0:
                envelope[-release_samples:] *= np.linspace(1, 0, release_samples)

            return filtered_noise * envelope


# ============================================================================
# Text-to-Speech Integration
# ============================================================================

class TTSEngine:
    """
    Text-to-Speech engine for converting text to speech audio.

    Supports multiple backends:
    - espeak: Fast, robotic, works offline
    - pyttsx3: Cross-platform, works offline
    - gtts: Google TTS, requires internet but sounds natural
    """

    def __init__(self, backend: str = 'espeak', sample_rate: int = SAMPLE_RATE):
        self.backend = backend
        self.sample_rate = sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        """
        Convert text to speech audio.

        Args:
            text: Text to speak

        Returns:
            Audio as numpy array
        """
        if self.backend == 'espeak':
            return self._espeak(text)
        elif self.backend == 'pyttsx3':
            return self._pyttsx3(text)
        elif self.backend == 'gtts':
            return self._gtts(text)
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

    def _espeak(self, text: str) -> np.ndarray:
        """Use espeak for TTS (Linux/Mac)."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            # Run espeak to generate WAV
            subprocess.run(
                [
                    'espeak',
                    '-w', temp_path,
                    '-s', '150',  # Speed
                    '-p', '50',   # Pitch
                    text
                ],
                check=True,
                capture_output=True
            )

            # Read the WAV file
            sr, audio = wavfile.read(temp_path)

            # Convert to float
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0

            # Resample if needed
            if sr != self.sample_rate:
                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = signal.resample(audio, num_samples)

            return audio

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _pyttsx3(self, text: str) -> np.ndarray:
        """Use pyttsx3 for TTS."""
        try:
            import pyttsx3
        except ImportError:
            raise ImportError("pyttsx3 not installed. Run: pip install pyttsx3")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.save_to_file(text, temp_path)
            engine.runAndWait()

            sr, audio = wavfile.read(temp_path)

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            if sr != self.sample_rate:
                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = signal.resample(audio, num_samples)

            return audio

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _gtts(self, text: str) -> np.ndarray:
        """Use Google TTS (requires internet)."""
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError("gtts not installed. Run: pip install gtts")

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            mp3_path = f.name

        wav_path = mp3_path.replace('.mp3', '.wav')

        try:
            # Generate MP3
            tts = gTTS(text=text, lang='en')
            tts.save(mp3_path)

            # Convert to WAV using ffmpeg
            subprocess.run(
                ['ffmpeg', '-y', '-i', mp3_path, '-ar', str(self.sample_rate), wav_path],
                check=True,
                capture_output=True
            )

            sr, audio = wavfile.read(wav_path)

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            # Handle stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            return audio

        finally:
            for path in [mp3_path, wav_path]:
                if os.path.exists(path):
                    os.unlink(path)


# ============================================================================
# Main R2D2 Voice Class
# ============================================================================

class R2D2Voice:
    """
    Main interface for R2-D2 voice synthesis.

    Can convert either:
    - Text -> R2D2 sounds (via TTS)
    - Speech audio -> R2D2 sounds (direct conversion)

    Example usage:
        voice = R2D2Voice()

        # From text
        audio = voice.text_to_r2d2("Hello there!")
        voice.save("hello.wav", audio)

        # From speech audio
        speech = voice.load("my_voice.wav")
        r2d2 = voice.speech_to_r2d2(speech)
        voice.save("r2d2_output.wav", r2d2)
    """

    def __init__(
        self,
        tts_backend: str = 'espeak',
        expressiveness: float = 1.5,
        chirpiness: float = 0.3,
        noise_blend: float = 0.1,
        sample_rate: int = SAMPLE_RATE,
    ):
        """
        Initialize R2D2 voice synthesizer.

        Args:
            tts_backend: TTS engine to use ('espeak', 'pyttsx3', 'gtts')
            expressiveness: How much to exaggerate pitch movements (1.0 = natural)
            chirpiness: Probability of adding frequency sweeps (0-1)
            noise_blend: Amount of noise to blend into beeps (0-1)
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.extractor = ProsodyExtractor(sample_rate=sample_rate)
        self.synthesizer = R2D2Synthesizer(
            sample_rate=sample_rate,
            expressiveness=expressiveness,
            chirpiness=chirpiness,
            noise_blend=noise_blend,
        )
        self.tts = TTSEngine(backend=tts_backend, sample_rate=sample_rate)

    def text_to_r2d2(self, text: str) -> np.ndarray:
        """
        Convert text to R2-D2 sounds.

        Args:
            text: Text to convert

        Returns:
            R2D2 audio as numpy array
        """
        # First, convert text to speech
        speech = self.tts.synthesize(text)

        # Then convert speech to R2D2
        return self.speech_to_r2d2(speech)

    def speech_to_r2d2(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert speech audio to R2-D2 sounds.

        Args:
            audio: Speech audio as numpy array

        Returns:
            R2D2 audio as numpy array
        """
        # Extract prosodic features
        prosody = self.extractor.extract(audio)

        # Synthesize R2D2 sounds
        return self.synthesizer.synthesize(prosody)

    def load(self, path: str) -> np.ndarray:
        """Load audio from WAV file."""
        sr, audio = wavfile.read(path)

        # Convert to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128) / 128.0

        # Handle stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = signal.resample(audio, num_samples)

        return audio

    def save(self, path: str, audio: np.ndarray):
        """Save audio to WAV file."""
        # Convert to int16
        audio_int = (audio * 32767).astype(np.int16)
        wavfile.write(path, self.sample_rate, audio_int)

    def play(self, audio: np.ndarray):
        """Play audio (requires aplay on Linux or afplay on Mac)."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            self.save(temp_path, audio)

            # Try different audio players
            for player in ['aplay', 'afplay', 'paplay']:
                try:
                    subprocess.run([player, temp_path], check=True, capture_output=True)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue

            print(f"Could not play audio. Saved to: {temp_path}")

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# ============================================================================
# Convenience Functions
# ============================================================================

def text_to_r2d2(text: str, output_path: Optional[str] = None, **kwargs) -> np.ndarray:
    """
    Quick function to convert text to R2D2 sounds.

    Args:
        text: Text to convert
        output_path: Optional path to save WAV file
        **kwargs: Arguments passed to R2D2Voice

    Returns:
        R2D2 audio as numpy array
    """
    voice = R2D2Voice(**kwargs)
    audio = voice.text_to_r2d2(text)

    if output_path:
        voice.save(output_path, audio)

    return audio


def speech_to_r2d2(
    input_path: str,
    output_path: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    Quick function to convert speech audio to R2D2 sounds.

    Args:
        input_path: Path to input WAV file
        output_path: Optional path to save output WAV file
        **kwargs: Arguments passed to R2D2Voice

    Returns:
        R2D2 audio as numpy array
    """
    voice = R2D2Voice(**kwargs)
    speech = voice.load(input_path)
    audio = voice.speech_to_r2d2(speech)

    if output_path:
        voice.save(output_path, audio)

    return audio


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='R2-D2 Voice Synthesizer')
    parser.add_argument('--text', '-t', type=str, help='Text to convert to R2D2')
    parser.add_argument('--input', '-i', type=str, help='Input speech WAV file')
    parser.add_argument('--output', '-o', type=str, default='r2d2_output.wav',
                        help='Output WAV file')
    parser.add_argument('--tts', type=str, default='espeak',
                        choices=['espeak', 'pyttsx3', 'gtts'],
                        help='TTS backend to use')
    parser.add_argument('--expressiveness', type=float, default=1.5,
                        help='Pitch expressiveness (1.0 = natural)')
    parser.add_argument('--chirpiness', type=float, default=0.3,
                        help='Probability of frequency sweeps (0-1)')
    parser.add_argument('--play', '-p', action='store_true',
                        help='Play the output audio')

    args = parser.parse_args()

    voice = R2D2Voice(
        tts_backend=args.tts,
        expressiveness=args.expressiveness,
        chirpiness=args.chirpiness,
    )

    if args.text:
        print(f"Converting text: '{args.text}'")
        audio = voice.text_to_r2d2(args.text)
    elif args.input:
        print(f"Converting speech from: {args.input}")
        speech = voice.load(args.input)
        audio = voice.speech_to_r2d2(speech)
    else:
        # Demo mode
        print("Demo mode - generating sample R2D2 sounds...")
        audio = voice.text_to_r2d2("Hello! I am R2-D2. Nice to meet you!")

    voice.save(args.output, audio)
    print(f"Saved to: {args.output}")

    if args.play:
        print("Playing...")
        voice.play(audio)
