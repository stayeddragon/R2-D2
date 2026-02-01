"""
R2-D2 Enhanced Voice Synthesis

Full-featured R2D2 voice with:
- Emotion detection and modulation
- Multiple synthesis modes
- Characteristic R2D2 sound patterns
- Real-time audio streaming support
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from typing import Optional, List, Tuple, Generator
import tempfile
import subprocess
import os

from r2d2_voice import (
    ProsodyExtractor, ProsodyFrame, R2D2Segment,
    TTSEngine, SAMPLE_RATE
)
from r2d2_emotions import (
    EmotionProfile, EmotionModulator, EMOTIONS,
    detect_emotion_from_text, list_emotions
)


class EnhancedR2D2Synthesizer:
    """
    Enhanced R2D2 synthesizer with emotion support and
    more characteristic sound generation.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        emotion: str = 'neutral',
    ):
        self.sample_rate = sample_rate
        self.emotion_modulator = EmotionModulator(emotion)

    def set_emotion(self, emotion: str):
        """Set the current emotion."""
        self.emotion_modulator.set_emotion(emotion)

    def synthesize(self, prosody_frames: List[ProsodyFrame]) -> np.ndarray:
        """Synthesize R2D2 audio from prosody frames."""
        if not prosody_frames:
            return np.array([], dtype=np.float32)

        profile = self.emotion_modulator.get_profile()

        # Convert prosody to segments
        segments = self._prosody_to_segments(prosody_frames, profile)

        # Synthesize
        total_duration = prosody_frames[-1].time + 0.025
        # Apply tempo
        total_duration /= profile.tempo
        total_samples = int(total_duration * self.sample_rate)
        audio = np.zeros(total_samples, dtype=np.float64)

        for segment in segments:
            segment_audio = self._synthesize_segment(segment, profile)
            start_sample = int(segment.start_time * self.sample_rate / profile.tempo)
            end_sample = start_sample + len(segment_audio)

            if end_sample > len(audio):
                end_sample = len(audio)
                segment_audio = segment_audio[:end_sample - start_sample]

            if start_sample < len(audio) and len(segment_audio) > 0:
                audio[start_sample:end_sample] += segment_audio

        # Normalize with soft clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        audio = np.tanh(audio * 1.5) / np.tanh(1.5)

        return audio.astype(np.float32)

    def _prosody_to_segments(
        self,
        frames: List[ProsodyFrame],
        profile: EmotionProfile
    ) -> List[R2D2Segment]:
        """Convert prosody frames to R2D2 segments."""
        segments = []
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
                seg = self._frames_to_segment(current_group, current_type, profile)
                if seg is not None:
                    segments.append(seg)
                current_group = []

            current_type = frame_type
            current_group.append(frame)

        if current_group:
            seg = self._frames_to_segment(current_group, current_type, profile)
            if seg is not None:
                segments.append(seg)

        return segments

    def _frames_to_segment(
        self,
        frames: List[ProsodyFrame],
        segment_type: str,
        profile: EmotionProfile
    ) -> Optional[R2D2Segment]:
        """Convert frame group to segment with emotion modulation."""
        if segment_type == 'silence':
            return None

        start_time = frames[0].time
        duration = frames[-1].time - frames[0].time + 0.01

        if duration < 0.01:
            return None

        num_samples = int(duration * self.sample_rate)

        if segment_type == 'voiced':
            # Get pitch contour
            pitches = np.array([f.pitch if f.pitch > 0 else 150 for f in frames])

            # Normalize to 0-1
            pitch_min, pitch_max = 75, 400
            pitch_norm = (pitches - pitch_min) / (pitch_max - pitch_min)
            pitch_norm = np.clip(pitch_norm, 0, 1)

            # Apply expressiveness
            center = 0.5
            pitch_norm = center + (pitch_norm - center) * profile.expressiveness
            pitch_norm = np.clip(pitch_norm, 0, 1)

            # Map to R2D2 frequency range with emotion modulation
            freq_min = 200 * profile.freq_range + profile.freq_shift
            freq_max = 4000 * profile.freq_range + profile.freq_shift
            frequencies = freq_min + pitch_norm * (freq_max - freq_min)

            # Center around emotion's center frequency
            current_center = np.mean(frequencies)
            frequencies = frequencies + (profile.freq_center - current_center) * 0.5

            # Interpolate to sample rate
            frequencies = np.interp(
                np.linspace(0, 1, num_samples),
                np.linspace(0, 1, len(frequencies)),
                frequencies
            )

            # Add wobble/vibrato
            t = np.arange(num_samples) / self.sample_rate
            wobble = np.sin(2 * np.pi * profile.wobble_rate * t) * profile.wobble_depth
            frequencies = frequencies + wobble

            # Add chirp based on emotion
            if np.random.random() < profile.chirpiness:
                chirp_amount = np.random.uniform(100, 300)
                if profile.chirp_direction != 0:
                    chirp_amount *= np.sign(profile.chirp_direction)
                elif np.random.random() > 0.5:
                    chirp_amount *= -1
                chirp = np.linspace(0, chirp_amount, num_samples)
                frequencies = frequencies + chirp

            # Amplitude from energy
            energies = np.array([f.energy for f in frames])
            amplitudes = np.interp(
                np.linspace(0, 1, num_samples),
                np.linspace(0, 1, len(energies)),
                energies
            )

            # Smooth
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
            # Noise burst
            freq = np.random.uniform(400, 1200) + profile.freq_shift
            frequencies = np.full(num_samples, freq)

            energies = np.array([f.energy for f in frames])
            amplitudes = np.interp(
                np.linspace(0, 1, num_samples),
                np.linspace(0, 1, len(energies)),
                energies
            ) * 0.5

            return R2D2Segment(
                start_time=start_time,
                duration=duration,
                segment_type='noise',
                frequencies=frequencies,
                amplitudes=amplitudes
            )

    def _synthesize_segment(
        self,
        segment: R2D2Segment,
        profile: EmotionProfile
    ) -> np.ndarray:
        """Synthesize audio for a segment with emotion characteristics."""
        num_samples = len(segment.frequencies)
        t = np.arange(num_samples) / self.sample_rate

        if segment.segment_type == 'beep':
            # FM synthesis
            phase = 2 * np.pi * np.cumsum(segment.frequencies) / self.sample_rate

            # Waveform mixing based on emotion
            sine = np.sin(phase)
            squared = np.tanh(3 * np.sin(phase))
            tone = (1 - profile.square_mix) * sine + profile.square_mix * squared

            # Harmonics
            tone += profile.harmonic2 * np.sin(2 * phase)
            tone += profile.harmonic3 * np.sin(3 * phase)

            # Noise blend
            noise = np.random.randn(num_samples) * profile.noise_blend
            tone = tone + noise

            # Envelope
            envelope = segment.amplitudes.copy()

            # Attack/release from emotion
            attack_samples = max(1, int(profile.attack * self.sample_rate))
            attack_samples = min(attack_samples, num_samples // 4)
            if attack_samples > 0:
                envelope[:attack_samples] *= np.linspace(0, 1, attack_samples)

            release_samples = max(1, int(profile.release * self.sample_rate))
            release_samples = min(release_samples, num_samples // 4)
            if release_samples > 0:
                envelope[-release_samples:] *= np.linspace(1, 0, release_samples)

            # Staccato effect
            if profile.staccato > 0 and num_samples > 50:
                staccato_envelope = np.ones(num_samples)
                gap_size = int(num_samples * 0.1 * profile.staccato)
                if gap_size > 0:
                    for i in range(3, num_samples - gap_size, gap_size * 3):
                        fade = np.linspace(1, 0.2, gap_size)
                        end_idx = min(i + gap_size, num_samples)
                        staccato_envelope[i:end_idx] = fade[:end_idx - i]
                envelope = envelope * staccato_envelope

            # Tremolo
            if profile.tremolo_rate > 0:
                tremolo = 1 - profile.tremolo_depth * (0.5 + 0.5 * np.sin(
                    2 * np.pi * profile.tremolo_rate * t
                ))
                envelope = envelope * tremolo

            return tone * envelope

        else:  # noise
            noise = np.random.randn(num_samples)

            # Bandpass
            center_freq = np.mean(segment.frequencies)
            bandwidth = 400
            low = max(100, center_freq - bandwidth)
            high = min(self.sample_rate / 2 - 100, center_freq + bandwidth)

            try:
                sos = signal.butter(2, [low, high], btype='band',
                                   fs=self.sample_rate, output='sos')
                filtered = signal.sosfilt(sos, noise)
            except ValueError:
                filtered = noise

            # Envelope
            envelope = segment.amplitudes.copy()
            attack = max(1, int(0.003 * self.sample_rate))
            attack = min(attack, num_samples // 4)
            if attack > 0:
                envelope[:attack] *= np.linspace(0, 1, attack)
            release = max(1, int(0.005 * self.sample_rate))
            release = min(release, num_samples // 4)
            if release > 0:
                envelope[-release:] *= np.linspace(1, 0, release)

            return filtered * envelope * (1 + profile.noise_blend)


class R2D2VoiceEnhanced:
    """
    Full-featured R2D2 voice with emotion support.

    Example:
        voice = R2D2VoiceEnhanced()

        # Auto-detect emotion from text
        audio = voice.speak("I'm so happy to see you!")

        # Manual emotion
        audio = voice.speak("Warning! Danger ahead!", emotion='urgent')

        # From speech
        voice.convert_speech("input.wav", "output.wav", emotion='excited')
    """

    def __init__(
        self,
        tts_backend: str = 'espeak',
        default_emotion: str = 'neutral',
        auto_detect_emotion: bool = True,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.sample_rate = sample_rate
        self.extractor = ProsodyExtractor(sample_rate=sample_rate)
        self.synthesizer = EnhancedR2D2Synthesizer(
            sample_rate=sample_rate,
            emotion=default_emotion
        )
        self.tts = TTSEngine(backend=tts_backend, sample_rate=sample_rate)
        self.auto_detect = auto_detect_emotion

    def speak(
        self,
        text: str,
        emotion: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Convert text to R2D2 speech.

        Args:
            text: Text to speak
            emotion: Emotion to use (auto-detected if None)
            output_path: Optional path to save audio

        Returns:
            Audio as numpy array
        """
        # Detect or set emotion
        if emotion is None and self.auto_detect:
            emotion = detect_emotion_from_text(text)
        elif emotion is None:
            emotion = 'neutral'

        self.synthesizer.set_emotion(emotion)

        # TTS -> Prosody -> R2D2
        speech = self.tts.synthesize(text)
        prosody = self.extractor.extract(speech)
        audio = self.synthesizer.synthesize(prosody)

        if output_path:
            self.save(output_path, audio)

        return audio

    def convert_speech(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        emotion: str = 'neutral'
    ) -> np.ndarray:
        """
        Convert speech audio file to R2D2.

        Args:
            input_path: Path to input WAV
            output_path: Optional output path
            emotion: Emotion to apply

        Returns:
            Audio as numpy array
        """
        self.synthesizer.set_emotion(emotion)

        speech = self.load(input_path)
        prosody = self.extractor.extract(speech)
        audio = self.synthesizer.synthesize(prosody)

        if output_path:
            self.save(output_path, audio)

        return audio

    def load(self, path: str) -> np.ndarray:
        """Load audio from file."""
        sr, audio = wavfile.read(path)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if sr != self.sample_rate:
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = signal.resample(audio, num_samples)

        return audio

    def save(self, path: str, audio: np.ndarray):
        """Save audio to WAV file."""
        audio_int = (audio * 32767).astype(np.int16)
        wavfile.write(path, self.sample_rate, audio_int)

    def play(self, audio: np.ndarray):
        """Play audio."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            self.save(temp_path, audio)
            for player in ['aplay', 'afplay', 'paplay']:
                try:
                    subprocess.run([player, temp_path], check=True, capture_output=True)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def generate_beep(
        self,
        duration: float = 0.3,
        emotion: str = 'neutral'
    ) -> np.ndarray:
        """
        Generate a single R2D2 beep.

        Useful for acknowledgments, alerts, etc.
        """
        self.synthesizer.set_emotion(emotion)
        profile = self.synthesizer.emotion_modulator.get_profile()

        num_samples = int(duration * self.sample_rate)

        # Generate simple beep
        freq = profile.freq_center
        t = np.arange(num_samples) / self.sample_rate

        # Frequency with wobble
        frequencies = freq + profile.wobble_depth * np.sin(
            2 * np.pi * profile.wobble_rate * t
        )

        # Add chirp
        if np.random.random() < profile.chirpiness:
            chirp = np.linspace(0, 200 * profile.chirp_direction, num_samples)
            frequencies = frequencies + chirp

        # Synthesize
        phase = 2 * np.pi * np.cumsum(frequencies) / self.sample_rate
        tone = np.sin(phase)
        tone += profile.harmonic2 * np.sin(2 * phase)

        # Envelope
        envelope = np.ones(num_samples)
        attack = int(profile.attack * self.sample_rate)
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)
        release = int(profile.release * self.sample_rate)
        if release > 0:
            envelope[-release:] = np.linspace(1, 0, release)

        audio = (tone * envelope).astype(np.float32)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio

    def generate_sequence(
        self,
        pattern: str = 'random',
        length: int = 5,
        emotion: str = 'neutral'
    ) -> np.ndarray:
        """
        Generate a sequence of R2D2 beeps.

        Patterns:
        - 'random': Random beeps
        - 'ascending': Rising pitch
        - 'descending': Falling pitch
        - 'question': Ends with rising tone
        - 'exclaim': Ends with emphasis
        """
        self.synthesizer.set_emotion(emotion)
        profile = self.synthesizer.emotion_modulator.get_profile()

        segments = []

        for i in range(length):
            # Duration varies
            dur = np.random.uniform(0.1, 0.3)
            num_samples = int(dur * self.sample_rate)

            # Frequency based on pattern
            if pattern == 'ascending':
                base_freq = profile.freq_center + (i / length) * 800
            elif pattern == 'descending':
                base_freq = profile.freq_center + ((length - i) / length) * 800
            elif pattern == 'question':
                if i == length - 1:
                    base_freq = profile.freq_center + 600
                else:
                    base_freq = profile.freq_center + np.random.uniform(-200, 200)
            elif pattern == 'exclaim':
                if i == length - 1:
                    base_freq = profile.freq_center
                    dur *= 1.5
                    num_samples = int(dur * self.sample_rate)
                else:
                    base_freq = profile.freq_center + np.random.uniform(-100, 300)
            else:  # random
                base_freq = profile.freq_center + np.random.uniform(-400, 800)

            t = np.arange(num_samples) / self.sample_rate
            freq = base_freq + profile.wobble_depth * np.sin(
                2 * np.pi * profile.wobble_rate * t
            )

            phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
            tone = np.sin(phase)
            tone += profile.harmonic2 * np.sin(2 * phase)

            # Envelope
            envelope = np.ones(num_samples)
            attack = max(1, int(profile.attack * self.sample_rate))
            envelope[:attack] = np.linspace(0, 1, attack)
            release = max(1, int(profile.release * self.sample_rate))
            envelope[-release:] = np.linspace(1, 0, release)

            segments.append(tone * envelope)

            # Gap between beeps
            gap = int(np.random.uniform(0.02, 0.08) * self.sample_rate)
            segments.append(np.zeros(gap))

        audio = np.concatenate(segments).astype(np.float32)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio


# Convenience functions

def speak(text: str, emotion: Optional[str] = None, **kwargs) -> np.ndarray:
    """Quick function to speak text as R2D2."""
    voice = R2D2VoiceEnhanced(**kwargs)
    return voice.speak(text, emotion=emotion)


def convert(input_path: str, output_path: str, emotion: str = 'neutral', **kwargs):
    """Quick function to convert speech to R2D2."""
    voice = R2D2VoiceEnhanced(**kwargs)
    voice.convert_speech(input_path, output_path, emotion=emotion)


def beep(emotion: str = 'neutral', **kwargs) -> np.ndarray:
    """Generate a single R2D2 beep."""
    voice = R2D2VoiceEnhanced(**kwargs)
    return voice.generate_beep(emotion=emotion)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='R2-D2 Enhanced Voice')
    parser.add_argument('--text', '-t', type=str, help='Text to speak')
    parser.add_argument('--input', '-i', type=str, help='Input WAV file')
    parser.add_argument('--output', '-o', type=str, default='r2d2_output.wav')
    parser.add_argument('--emotion', '-e', type=str, default=None,
                        help=f'Emotion: {", ".join(list_emotions())}')
    parser.add_argument('--list-emotions', action='store_true',
                        help='List available emotions')
    parser.add_argument('--beep', action='store_true', help='Generate single beep')
    parser.add_argument('--sequence', type=str, choices=[
                        'random', 'ascending', 'descending', 'question', 'exclaim'
                        ], help='Generate beep sequence')
    parser.add_argument('--play', '-p', action='store_true', help='Play output')
    parser.add_argument('--tts', type=str, default='espeak',
                        choices=['espeak', 'pyttsx3', 'gtts'])

    args = parser.parse_args()

    if args.list_emotions:
        print("Available emotions:")
        for em in list_emotions():
            from r2d2_emotions import get_emotion_description
            print(f"  {em}: {get_emotion_description(em)}")
        exit(0)

    voice = R2D2VoiceEnhanced(tts_backend=args.tts)

    if args.beep:
        audio = voice.generate_beep(emotion=args.emotion or 'neutral')
        print(f"Generated beep with emotion: {args.emotion or 'neutral'}")

    elif args.sequence:
        audio = voice.generate_sequence(
            pattern=args.sequence,
            emotion=args.emotion or 'neutral'
        )
        print(f"Generated {args.sequence} sequence")

    elif args.text:
        detected = detect_emotion_from_text(args.text) if not args.emotion else args.emotion
        audio = voice.speak(args.text, emotion=args.emotion)
        print(f"Converted: '{args.text}'")
        print(f"Emotion: {detected}")

    elif args.input:
        audio = voice.convert_speech(args.input, emotion=args.emotion or 'neutral')
        print(f"Converted: {args.input}")

    else:
        # Demo
        print("Demo mode - generating samples...")
        audio = voice.speak("Hello! I am R2-D2!")
        print("Generated demo with auto-detected emotion")

    voice.save(args.output, audio)
    print(f"Saved: {args.output}")

    if args.play:
        voice.play(audio)
