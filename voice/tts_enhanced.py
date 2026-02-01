"""
Enhanced TTS with Prosody Control

Since basic TTS (like espeak) produces flat, robotic speech,
we need ways to inject expressiveness:

1. Prosody Injection - Add artificial pitch/energy contours
2. Better TTS backends - Support for neural TTS
3. Tuning parameters - Control how expressive the output is
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
import tempfile
import subprocess
import os
import json

from r2d2_voice import SAMPLE_RATE, ProsodyFrame


@dataclass
class ProsodyContour:
    """Defines an artificial prosody pattern to apply."""
    name: str
    pitch_curve: Callable[[float], float]  # t (0-1) -> pitch multiplier
    energy_curve: Callable[[float], float]  # t (0-1) -> energy multiplier


# Predefined prosody patterns
PROSODY_PATTERNS = {
    'flat': ProsodyContour(
        name='flat',
        pitch_curve=lambda t: 1.0,
        energy_curve=lambda t: 1.0,
    ),

    'excited': ProsodyContour(
        name='excited',
        pitch_curve=lambda t: 1.0 + 0.3 * np.sin(t * 4 * np.pi) + 0.2 * t,
        energy_curve=lambda t: 0.8 + 0.4 * np.abs(np.sin(t * 3 * np.pi)),
    ),

    'sad': ProsodyContour(
        name='sad',
        pitch_curve=lambda t: 1.0 - 0.2 * t - 0.1 * np.sin(t * 2 * np.pi),
        energy_curve=lambda t: 0.9 - 0.3 * t,
    ),

    'question': ProsodyContour(
        name='question',
        pitch_curve=lambda t: 1.0 + 0.4 * (t ** 2),  # Rising at end
        energy_curve=lambda t: 0.8 + 0.2 * t,
    ),

    'statement': ProsodyContour(
        name='statement',
        pitch_curve=lambda t: 1.0 - 0.15 * t,  # Slightly falling
        energy_curve=lambda t: 1.0 - 0.1 * t,
    ),

    'exclamation': ProsodyContour(
        name='exclamation',
        pitch_curve=lambda t: 1.2 - 0.3 * t + 0.2 * np.sin(t * 6 * np.pi),
        energy_curve=lambda t: 1.2 * np.exp(-t * 0.5),
    ),

    'urgent': ProsodyContour(
        name='urgent',
        pitch_curve=lambda t: 1.1 + 0.2 * np.sin(t * 8 * np.pi),
        energy_curve=lambda t: 1.0 + 0.3 * np.abs(np.sin(t * 6 * np.pi)),
    ),

    'playful': ProsodyContour(
        name='playful',
        pitch_curve=lambda t: 1.0 + 0.25 * np.sin(t * 5 * np.pi) * np.sin(t * 13 * np.pi),
        energy_curve=lambda t: 0.9 + 0.2 * np.random.random(),
    ),

    'whisper': ProsodyContour(
        name='whisper',
        pitch_curve=lambda t: 0.9 + 0.05 * np.sin(t * 3 * np.pi),
        energy_curve=lambda t: 0.4 + 0.1 * np.sin(t * 2 * np.pi),
    ),

    'dramatic': ProsodyContour(
        name='dramatic',
        pitch_curve=lambda t: 1.0 + 0.4 * np.sin(t * 2 * np.pi),
        energy_curve=lambda t: 0.6 + 0.6 * np.abs(np.sin(t * 2 * np.pi)),
    ),
}


class ProsodyInjector:
    """
    Injects artificial prosody into flat prosody frames.

    This compensates for robotic TTS by adding natural-sounding
    pitch and energy variations.
    """

    def __init__(
        self,
        base_pitch: float = 200,      # Base pitch in Hz
        pitch_range: float = 100,      # Pitch variation range
        energy_base: float = 0.7,      # Base energy level
        energy_range: float = 0.3,     # Energy variation range
    ):
        self.base_pitch = base_pitch
        self.pitch_range = pitch_range
        self.energy_base = energy_base
        self.energy_range = energy_range

    def inject(
        self,
        frames: List[ProsodyFrame],
        pattern: str = 'statement',
        intensity: float = 1.0,
    ) -> List[ProsodyFrame]:
        """
        Inject prosody pattern into frames.

        Args:
            frames: Original prosody frames
            pattern: Prosody pattern name
            intensity: How strongly to apply (0-2, 1=normal)

        Returns:
            Modified prosody frames
        """
        if pattern not in PROSODY_PATTERNS:
            pattern = 'statement'

        contour = PROSODY_PATTERNS[pattern]

        # Find voiced frames only
        voiced_indices = [i for i, f in enumerate(frames) if f.is_voiced]

        if not voiced_indices:
            return frames

        # Create modified frames
        new_frames = []

        for i, frame in enumerate(frames):
            if frame.is_silence:
                new_frames.append(frame)
                continue

            # Calculate position in utterance (0-1)
            if voiced_indices:
                pos = voiced_indices.index(i) / len(voiced_indices) if i in voiced_indices else 0.5
            else:
                pos = 0.5

            # Get contour values
            pitch_mult = contour.pitch_curve(pos)
            energy_mult = contour.energy_curve(pos)

            # Apply intensity
            pitch_mult = 1.0 + (pitch_mult - 1.0) * intensity
            energy_mult = 1.0 + (energy_mult - 1.0) * intensity

            # Calculate new values
            if frame.is_voiced and frame.pitch > 0:
                new_pitch = frame.pitch * pitch_mult
            else:
                # Generate synthetic pitch for unvoiced
                new_pitch = self.base_pitch + self.pitch_range * (pitch_mult - 1.0)

            new_energy = max(0, min(1, frame.energy * energy_mult))

            new_frames.append(ProsodyFrame(
                time=frame.time,
                pitch=new_pitch,
                energy=new_energy,
                is_voiced=True,  # Treat all as voiced for R2D2
                is_silence=False,
            ))

        return new_frames

    def generate_synthetic(
        self,
        duration: float,
        pattern: str = 'statement',
        frame_rate: float = 100,  # frames per second
    ) -> List[ProsodyFrame]:
        """
        Generate fully synthetic prosody frames.

        Useful when you don't have TTS at all and just want
        to generate R2D2 sounds from text length/pattern.
        """
        if pattern not in PROSODY_PATTERNS:
            pattern = 'statement'

        contour = PROSODY_PATTERNS[pattern]

        num_frames = int(duration * frame_rate)
        frames = []

        for i in range(num_frames):
            t = i / num_frames
            time = i / frame_rate

            pitch_mult = contour.pitch_curve(t)
            energy_mult = contour.energy_curve(t)

            pitch = self.base_pitch + self.pitch_range * (pitch_mult - 0.5)
            energy = self.energy_base + self.energy_range * (energy_mult - 0.5)

            # Add some natural variation
            pitch += np.random.randn() * 10
            energy += np.random.randn() * 0.05

            frames.append(ProsodyFrame(
                time=time,
                pitch=max(50, pitch),
                energy=max(0, min(1, energy)),
                is_voiced=True,
                is_silence=False,
            ))

        return frames


class EnhancedTTS:
    """
    Enhanced TTS with multiple backends and prosody control.

    Backends:
    - espeak: Fast, offline, robotic (default)
    - pyttsx3: Cross-platform, offline
    - gtts: Google TTS, natural but needs internet
    - coqui: Neural TTS, very natural, offline (if installed)
    - piper: Fast neural TTS, offline (if installed)
    """

    def __init__(
        self,
        backend: str = 'espeak',
        sample_rate: int = SAMPLE_RATE,
        voice: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
    ):
        self.backend = backend
        self.sample_rate = sample_rate
        self.voice = voice
        self.speed = speed
        self.pitch = pitch

        # Check what's available
        self._available_backends = self._detect_backends()

        if backend not in self._available_backends:
            print(f"Warning: {backend} not available. Available: {self._available_backends}")
            if 'espeak' in self._available_backends:
                self.backend = 'espeak'
            elif self._available_backends:
                self.backend = self._available_backends[0]
            else:
                raise RuntimeError("No TTS backend available!")

    def _detect_backends(self) -> List[str]:
        """Detect which TTS backends are available."""
        available = []

        # Check espeak
        try:
            subprocess.run(['espeak', '--version'], capture_output=True, check=True)
            available.append('espeak')
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Check espeak-ng (better than espeak)
        try:
            subprocess.run(['espeak-ng', '--version'], capture_output=True, check=True)
            available.append('espeak-ng')
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Check pyttsx3
        try:
            import pyttsx3
            available.append('pyttsx3')
        except ImportError:
            pass

        # Check gtts
        try:
            from gtts import gTTS
            available.append('gtts')
        except ImportError:
            pass

        # Check piper (fast neural TTS)
        try:
            subprocess.run(['piper', '--version'], capture_output=True, check=True)
            available.append('piper')
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        return available

    def list_backends(self) -> List[str]:
        """Return list of available backends."""
        return self._available_backends

    def synthesize(
        self,
        text: str,
        prosody_boost: Optional[str] = None,
        boost_intensity: float = 1.0,
    ) -> Tuple[np.ndarray, List[ProsodyFrame]]:
        """
        Synthesize speech and return audio + prosody frames.

        Args:
            text: Text to synthesize
            prosody_boost: Prosody pattern to inject ('excited', 'sad', etc.)
            boost_intensity: How strong to apply boost (0-2)

        Returns:
            Tuple of (audio array, prosody frames)
        """
        # Get raw audio from TTS
        audio = self._synth_audio(text)

        # Extract prosody
        from r2d2_voice import ProsodyExtractor
        extractor = ProsodyExtractor(sample_rate=self.sample_rate)
        frames = extractor.extract(audio)

        # Boost prosody if requested
        if prosody_boost:
            injector = ProsodyInjector()
            frames = injector.inject(frames, prosody_boost, boost_intensity)

        return audio, frames

    def _synth_audio(self, text: str) -> np.ndarray:
        """Synthesize audio using current backend."""
        if self.backend in ['espeak', 'espeak-ng']:
            return self._espeak(text)
        elif self.backend == 'pyttsx3':
            return self._pyttsx3(text)
        elif self.backend == 'gtts':
            return self._gtts(text)
        elif self.backend == 'piper':
            return self._piper(text)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _espeak(self, text: str) -> np.ndarray:
        """Use espeak/espeak-ng."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            cmd = self.backend  # 'espeak' or 'espeak-ng'
            args = [
                cmd,
                '-w', temp_path,
                '-s', str(int(150 * self.speed)),
                '-p', str(int(50 * self.pitch)),
            ]
            if self.voice:
                args.extend(['-v', self.voice])
            args.append(text)

            subprocess.run(args, check=True, capture_output=True)

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

    def _pyttsx3(self, text: str) -> np.ndarray:
        """Use pyttsx3."""
        import pyttsx3

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', int(150 * self.speed))

            if self.voice:
                voices = engine.getProperty('voices')
                for v in voices:
                    if self.voice.lower() in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break

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
        """Use Google TTS."""
        from gtts import gTTS

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            mp3_path = f.name
        wav_path = mp3_path.replace('.mp3', '.wav')

        try:
            tts = gTTS(text=text, lang='en')
            tts.save(mp3_path)

            subprocess.run([
                'ffmpeg', '-y', '-i', mp3_path,
                '-ar', str(self.sample_rate),
                wav_path
            ], check=True, capture_output=True)

            sr, audio = wavfile.read(wav_path)

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            return audio

        finally:
            for path in [mp3_path, wav_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def _piper(self, text: str) -> np.ndarray:
        """Use Piper neural TTS."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            args = ['piper', '--output_file', temp_path]
            if self.voice:
                args.extend(['--model', self.voice])

            proc = subprocess.run(
                args,
                input=text.encode(),
                check=True,
                capture_output=True
            )

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


def list_prosody_patterns() -> List[str]:
    """Return available prosody patterns."""
    return list(PROSODY_PATTERNS.keys())


def get_pattern_description(pattern: str) -> str:
    """Get description of a prosody pattern."""
    descriptions = {
        'flat': 'No variation - monotone',
        'excited': 'High energy, bouncy pitch',
        'sad': 'Falling pitch and energy',
        'question': 'Rising pitch at end',
        'statement': 'Slight falling pattern',
        'exclamation': 'High start, emphatic',
        'urgent': 'Rapid variations',
        'playful': 'Irregular, bouncy',
        'whisper': 'Low energy, subtle',
        'dramatic': 'Large pitch swings',
    }
    return descriptions.get(pattern, 'Unknown pattern')
