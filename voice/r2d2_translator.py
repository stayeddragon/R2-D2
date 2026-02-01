"""
R2-D2 Translator

Bidirectional translation between human language and R2D2 sounds:

Text/Speech -> R2D2 sounds (encoding)
R2D2 sounds -> Meaning/Emotion (decoding)

The reverse translation works by analyzing the prosodic features
of R2D2 sounds and inferring:
- Emotion/mood
- Utterance type (question, statement, exclamation)
- Rough semantic meaning based on patterns

Note: Exact text recovery isn't possible since R2D2 sounds encode
prosody, not phonemes. But we can infer the gist/emotion.
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from enum import Enum
import os
import tempfile
import subprocess

from r2d2_voice import SAMPLE_RATE


class UtteranceType(Enum):
    """Type of R2D2 utterance based on intonation."""
    QUESTION = "question"
    STATEMENT = "statement"
    EXCLAMATION = "exclamation"
    ACKNOWLEDGMENT = "acknowledgment"
    ALERT = "alert"
    GREETING = "greeting"
    NEGATIVE = "negative"


@dataclass
class R2D2Analysis:
    """Analysis result for R2D2 audio."""
    # Core metrics
    duration: float
    num_beeps: int
    avg_frequency: float
    freq_range: Tuple[float, float]
    avg_energy: float

    # Patterns
    pitch_direction: str  # 'rising', 'falling', 'flat', 'varied'
    tempo: str            # 'slow', 'medium', 'fast', 'urgent'
    complexity: str       # 'simple', 'moderate', 'complex'

    # Classifications
    emotion: str
    emotion_confidence: float
    utterance_type: UtteranceType

    # Interpretation
    interpretation: str
    alternative_interpretations: List[str]


class R2D2Analyzer:
    """
    Analyzes R2D2 audio to extract features and meaning.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    def analyze(self, audio: np.ndarray) -> R2D2Analysis:
        """
        Analyze R2D2 audio and extract meaning.

        Args:
            audio: R2D2 audio as numpy array

        Returns:
            R2D2Analysis with detected features and interpretation
        """
        # Ensure float and normalize
        audio = audio.astype(np.float64)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        duration = len(audio) / self.sample_rate

        # Extract frequency contour
        frequencies, times, energies = self._extract_features(audio)

        # Calculate metrics
        avg_freq = np.mean(frequencies[frequencies > 0]) if len(frequencies[frequencies > 0]) > 0 else 800
        freq_min = np.min(frequencies[frequencies > 0]) if len(frequencies[frequencies > 0]) > 0 else 200
        freq_max = np.max(frequencies) if len(frequencies) > 0 else 2000
        avg_energy = np.mean(energies)

        # Count beeps (segments separated by low energy)
        num_beeps = self._count_beeps(energies)

        # Analyze patterns
        pitch_direction = self._analyze_pitch_direction(frequencies)
        tempo = self._analyze_tempo(num_beeps, duration)
        complexity = self._analyze_complexity(frequencies, energies)

        # Classify emotion
        emotion, confidence = self._classify_emotion(
            avg_freq, freq_max - freq_min, avg_energy,
            pitch_direction, tempo, complexity
        )

        # Determine utterance type
        utterance_type = self._classify_utterance_type(
            pitch_direction, num_beeps, duration, emotion
        )

        # Generate interpretation
        interpretation, alternatives = self._generate_interpretation(
            emotion, utterance_type, num_beeps, duration
        )

        return R2D2Analysis(
            duration=duration,
            num_beeps=num_beeps,
            avg_frequency=avg_freq,
            freq_range=(freq_min, freq_max),
            avg_energy=avg_energy,
            pitch_direction=pitch_direction,
            tempo=tempo,
            complexity=complexity,
            emotion=emotion,
            emotion_confidence=confidence,
            utterance_type=utterance_type,
            interpretation=interpretation,
            alternative_interpretations=alternatives,
        )

    def _extract_features(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract frequency and energy contours from audio."""
        frame_size = int(0.025 * self.sample_rate)
        hop_size = int(0.010 * self.sample_rate)

        num_frames = (len(audio) - frame_size) // hop_size + 1

        frequencies = []
        times = []
        energies = []

        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_size
            frame = audio[start:end]

            time = (start + frame_size // 2) / self.sample_rate
            energy = np.sqrt(np.mean(frame ** 2))

            # Frequency detection via FFT peak
            if energy > 0.01:
                windowed = frame * np.hanning(len(frame))
                fft = np.fft.rfft(windowed)
                freqs = np.fft.rfftfreq(len(frame), 1/self.sample_rate)

                # Focus on R2D2 frequency range
                mask = (freqs > 150) & (freqs < 5000)
                if np.any(mask):
                    peak_idx = np.argmax(np.abs(fft[mask]))
                    freq = freqs[mask][peak_idx]
                else:
                    freq = 0
            else:
                freq = 0

            frequencies.append(freq)
            times.append(time)
            energies.append(energy)

        return np.array(frequencies), np.array(times), np.array(energies)

    def _count_beeps(self, energies: np.ndarray) -> int:
        """Count distinct beeps in the audio."""
        threshold = np.mean(energies) * 0.3

        # Find transitions
        is_beep = energies > threshold
        transitions = np.diff(is_beep.astype(int))

        # Count rising edges (start of beeps)
        num_beeps = np.sum(transitions == 1)

        # Handle edge case where audio starts with a beep
        if is_beep[0]:
            num_beeps += 1

        return max(1, num_beeps)

    def _analyze_pitch_direction(self, frequencies: np.ndarray) -> str:
        """Analyze overall pitch direction."""
        # Filter out silent frames
        valid = frequencies[frequencies > 0]

        if len(valid) < 3:
            return 'flat'

        # Compare first and last thirds
        third = len(valid) // 3
        if third < 1:
            return 'flat'

        start_avg = np.mean(valid[:third])
        end_avg = np.mean(valid[-third:])
        mid_avg = np.mean(valid[third:-third]) if len(valid) > 2*third else start_avg

        # Calculate variation
        variation = np.std(valid) / np.mean(valid) if np.mean(valid) > 0 else 0

        if variation > 0.3:
            return 'varied'
        elif end_avg > start_avg * 1.15:
            return 'rising'
        elif end_avg < start_avg * 0.85:
            return 'falling'
        else:
            return 'flat'

    def _analyze_tempo(self, num_beeps: int, duration: float) -> str:
        """Analyze the tempo/speed of the utterance."""
        if duration < 0.1:
            return 'fast'

        beeps_per_second = num_beeps / duration

        if beeps_per_second > 8:
            return 'urgent'
        elif beeps_per_second > 4:
            return 'fast'
        elif beeps_per_second > 2:
            return 'medium'
        else:
            return 'slow'

    def _analyze_complexity(
        self,
        frequencies: np.ndarray,
        energies: np.ndarray
    ) -> str:
        """Analyze how complex the utterance is."""
        valid_freq = frequencies[frequencies > 0]

        if len(valid_freq) < 2:
            return 'simple'

        # Frequency variation
        freq_variation = np.std(valid_freq) / np.mean(valid_freq) if np.mean(valid_freq) > 0 else 0

        # Energy variation
        energy_variation = np.std(energies) / np.mean(energies) if np.mean(energies) > 0 else 0

        # Number of direction changes in frequency
        if len(valid_freq) > 2:
            diffs = np.diff(valid_freq)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            change_ratio = sign_changes / len(diffs)
        else:
            change_ratio = 0

        complexity_score = freq_variation + energy_variation + change_ratio

        if complexity_score > 1.0:
            return 'complex'
        elif complexity_score > 0.4:
            return 'moderate'
        else:
            return 'simple'

    def _classify_emotion(
        self,
        avg_freq: float,
        freq_range: float,
        avg_energy: float,
        pitch_direction: str,
        tempo: str,
        complexity: str,
    ) -> Tuple[str, float]:
        """Classify the emotion from features."""
        # Score each emotion
        scores = {}

        # Happy: high freq, rising/varied, fast
        scores['happy'] = 0
        if avg_freq > 1000:
            scores['happy'] += 0.3
        if pitch_direction in ['rising', 'varied']:
            scores['happy'] += 0.2
        if tempo in ['fast', 'medium']:
            scores['happy'] += 0.2
        if avg_energy > 0.4:
            scores['happy'] += 0.1

        # Excited: very high freq, fast, varied
        scores['excited'] = 0
        if avg_freq > 1200:
            scores['excited'] += 0.3
        if tempo in ['fast', 'urgent']:
            scores['excited'] += 0.3
        if complexity == 'complex':
            scores['excited'] += 0.2
        if pitch_direction == 'varied':
            scores['excited'] += 0.2

        # Sad: low freq, falling, slow
        scores['sad'] = 0
        if avg_freq < 700:
            scores['sad'] += 0.3
        if pitch_direction == 'falling':
            scores['sad'] += 0.3
        if tempo == 'slow':
            scores['sad'] += 0.2
        if avg_energy < 0.3:
            scores['sad'] += 0.1

        # Worried: medium freq, varied, medium tempo
        scores['worried'] = 0
        if 600 < avg_freq < 900:
            scores['worried'] += 0.2
        if complexity in ['moderate', 'complex']:
            scores['worried'] += 0.2
        if pitch_direction == 'varied':
            scores['worried'] += 0.2
        if tempo == 'medium':
            scores['worried'] += 0.1

        # Curious: rising pitch, medium-high freq
        scores['curious'] = 0
        if pitch_direction == 'rising':
            scores['curious'] += 0.4
        if 800 < avg_freq < 1200:
            scores['curious'] += 0.2
        if complexity == 'moderate':
            scores['curious'] += 0.1

        # Urgent: fast/urgent tempo, high energy
        scores['urgent'] = 0
        if tempo == 'urgent':
            scores['urgent'] += 0.4
        if avg_energy > 0.5:
            scores['urgent'] += 0.2
        if complexity in ['moderate', 'complex']:
            scores['urgent'] += 0.1

        # Affirmative: simple, short, medium freq
        scores['affirmative'] = 0
        if complexity == 'simple':
            scores['affirmative'] += 0.3
        if 700 < avg_freq < 1000:
            scores['affirmative'] += 0.2
        if pitch_direction in ['flat', 'rising']:
            scores['affirmative'] += 0.1

        # Negative: low freq, falling, simple
        scores['negative'] = 0
        if avg_freq < 600:
            scores['negative'] += 0.3
        if pitch_direction == 'falling':
            scores['negative'] += 0.3
        if complexity == 'simple':
            scores['negative'] += 0.1

        # Neutral fallback
        scores['neutral'] = 0.2

        # Find best match
        best_emotion = max(scores, key=scores.get)
        best_score = scores[best_emotion]

        # Calculate confidence
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.5

        return best_emotion, min(1.0, confidence)

    def _classify_utterance_type(
        self,
        pitch_direction: str,
        num_beeps: int,
        duration: float,
        emotion: str,
    ) -> UtteranceType:
        """Classify the type of utterance."""
        if pitch_direction == 'rising':
            return UtteranceType.QUESTION

        if emotion == 'urgent':
            return UtteranceType.ALERT

        if emotion == 'negative':
            return UtteranceType.NEGATIVE

        if emotion in ['affirmative', 'happy'] and num_beeps <= 2 and duration < 0.5:
            return UtteranceType.ACKNOWLEDGMENT

        if emotion in ['happy', 'excited'] and duration < 0.8:
            return UtteranceType.GREETING

        if emotion in ['excited', 'urgent']:
            return UtteranceType.EXCLAMATION

        return UtteranceType.STATEMENT

    def _generate_interpretation(
        self,
        emotion: str,
        utterance_type: UtteranceType,
        num_beeps: int,
        duration: float,
    ) -> Tuple[str, List[str]]:
        """Generate human-readable interpretation."""
        # Interpretation templates
        interpretations = {
            ('happy', UtteranceType.GREETING): [
                "Hello there! Nice to see you!",
                "Hi! I'm happy to see you!",
                "Greetings, friend!",
            ],
            ('happy', UtteranceType.STATEMENT): [
                "I'm feeling good about this!",
                "That sounds great!",
                "I like that idea!",
            ],
            ('happy', UtteranceType.ACKNOWLEDGMENT): [
                "Yes, definitely!",
                "I agree!",
                "That's right!",
            ],
            ('excited', UtteranceType.EXCLAMATION): [
                "This is amazing!",
                "Wow, that's incredible!",
                "I can't believe it!",
            ],
            ('excited', UtteranceType.STATEMENT): [
                "I found something interesting!",
                "Look at this!",
                "You need to see this!",
            ],
            ('sad', UtteranceType.STATEMENT): [
                "I'm not feeling great about this...",
                "That's unfortunate...",
                "I miss the old days...",
            ],
            ('worried', UtteranceType.STATEMENT): [
                "I'm not sure about this...",
                "Something seems wrong...",
                "I have a bad feeling about this...",
            ],
            ('worried', UtteranceType.QUESTION): [
                "Are you sure about this?",
                "Is everything okay?",
                "What's happening?",
            ],
            ('curious', UtteranceType.QUESTION): [
                "What's that over there?",
                "Can you tell me more?",
                "What do you mean?",
            ],
            ('curious', UtteranceType.STATEMENT): [
                "That's interesting...",
                "Let me take a look...",
                "I wonder what this is...",
            ],
            ('urgent', UtteranceType.ALERT): [
                "Warning! Danger detected!",
                "Alert! We need to go now!",
                "Emergency! Take action!",
            ],
            ('urgent', UtteranceType.EXCLAMATION): [
                "Hurry! There's no time!",
                "Quick, follow me!",
                "We need to move!",
            ],
            ('affirmative', UtteranceType.ACKNOWLEDGMENT): [
                "Yes, understood.",
                "Affirmative.",
                "Got it.",
            ],
            ('affirmative', UtteranceType.STATEMENT): [
                "Yes, I can do that.",
                "Of course, right away.",
                "I'll take care of it.",
            ],
            ('negative', UtteranceType.NEGATIVE): [
                "No, I don't think so.",
                "That's not right.",
                "I disagree.",
            ],
            ('negative', UtteranceType.STATEMENT): [
                "I can't do that.",
                "That won't work.",
                "Something's wrong.",
            ],
            ('neutral', UtteranceType.STATEMENT): [
                "I understand.",
                "Processing...",
                "Let me think about that.",
            ],
        }

        key = (emotion, utterance_type)

        if key in interpretations:
            options = interpretations[key]
        else:
            # Fallback based on emotion
            fallbacks = {
                'happy': ["I'm happy!", "That's wonderful!", "Great!"],
                'excited': ["Wow!", "Amazing!", "Incredible!"],
                'sad': ["Oh no...", "That's sad...", "I see..."],
                'worried': ["Hmm...", "I'm not sure...", "Be careful..."],
                'curious': ["What's this?", "Interesting...", "Tell me more."],
                'urgent': ["Hurry!", "Alert!", "Quick!"],
                'affirmative': ["Yes.", "Okay.", "Understood."],
                'negative': ["No.", "I can't.", "That's wrong."],
                'neutral': ["I see.", "Okay.", "Processing..."],
            }
            options = fallbacks.get(emotion, ["Beep boop!"])

        # Select based on complexity (longer = first option)
        if duration > 1.0:
            idx = 0
        elif duration > 0.5:
            idx = min(1, len(options) - 1)
        else:
            idx = min(2, len(options) - 1)

        interpretation = options[idx]
        alternatives = [o for o in options if o != interpretation]

        return interpretation, alternatives


class R2D2Translator:
    """
    Bidirectional translator between human language and R2D2 sounds.

    Usage:
        translator = R2D2Translator()

        # Text to R2D2
        audio = translator.to_r2d2("Hello there!")
        translator.save("r2d2.wav", audio)

        # R2D2 to interpretation
        result = translator.from_r2d2("r2d2.wav")
        print(result.interpretation)  # "Hi! I'm happy to see you!"
    """

    def __init__(
        self,
        tts_backend: str = 'espeak',
        sample_rate: int = SAMPLE_RATE,
    ):
        self.sample_rate = sample_rate
        self.analyzer = R2D2Analyzer(sample_rate=sample_rate)

        # Import voice synthesizer
        from r2d2_enhanced import R2D2VoiceEnhanced
        self.voice = R2D2VoiceEnhanced(
            tts_backend=tts_backend,
            sample_rate=sample_rate,
        )

    def to_r2d2(
        self,
        text: str,
        emotion: Optional[str] = None,
    ) -> np.ndarray:
        """
        Translate text to R2D2 sounds.

        Args:
            text: Text to translate
            emotion: Emotion to convey (auto-detected if None)

        Returns:
            R2D2 audio as numpy array
        """
        return self.voice.speak(text, emotion=emotion)

    def from_r2d2(
        self,
        audio_or_path,
    ) -> R2D2Analysis:
        """
        Translate R2D2 sounds to meaning.

        Args:
            audio_or_path: Either numpy array or path to WAV file

        Returns:
            R2D2Analysis with interpretation and detected features
        """
        if isinstance(audio_or_path, str):
            audio = self._load_audio(audio_or_path)
        else:
            audio = audio_or_path

        return self.analyzer.analyze(audio)

    def translate_conversation(
        self,
        audio_or_path,
    ) -> List[R2D2Analysis]:
        """
        Translate a longer R2D2 audio with multiple utterances.

        Segments the audio and analyzes each segment.
        """
        if isinstance(audio_or_path, str):
            audio = self._load_audio(audio_or_path)
        else:
            audio = audio_or_path

        # Segment into individual utterances
        segments = self._segment_audio(audio)

        results = []
        for segment in segments:
            if len(segment) > int(0.05 * self.sample_rate):  # Skip very short
                analysis = self.analyzer.analyze(segment)
                results.append(analysis)

        return results

    def _load_audio(self, path: str) -> np.ndarray:
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

    def _segment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """Segment audio into individual utterances."""
        # Calculate envelope
        frame_size = int(0.02 * self.sample_rate)
        hop_size = frame_size // 2

        envelope = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            envelope.append(np.sqrt(np.mean(frame ** 2)))

        envelope = np.array(envelope)

        # Find silence threshold
        threshold = np.mean(envelope) * 0.2

        # Find segments
        is_sound = envelope > threshold
        transitions = np.diff(is_sound.astype(int))

        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]

        # Handle edge cases
        if is_sound[0]:
            starts = np.concatenate([[0], starts])
        if is_sound[-1]:
            ends = np.concatenate([ends, [len(is_sound) - 1]])

        # Convert to samples
        segments = []
        for start, end in zip(starts, ends):
            start_sample = start * hop_size
            end_sample = min((end + 1) * hop_size + frame_size, len(audio))

            # Add padding
            start_sample = max(0, start_sample - int(0.02 * self.sample_rate))
            end_sample = min(len(audio), end_sample + int(0.02 * self.sample_rate))

            segments.append(audio[start_sample:end_sample])

        return segments if segments else [audio]

    def save(self, path: str, audio: np.ndarray):
        """Save audio to file."""
        self.voice.save(path, audio)

    def play(self, audio: np.ndarray):
        """Play audio."""
        self.voice.play(audio)


def format_analysis(analysis: R2D2Analysis) -> str:
    """Format analysis result as readable string."""
    lines = [
        f"R2-D2 says: \"{analysis.interpretation}\"",
        f"",
        f"Emotion: {analysis.emotion} ({analysis.emotion_confidence:.0%} confidence)",
        f"Type: {analysis.utterance_type.value}",
        f"",
        f"Details:",
        f"  Duration: {analysis.duration:.2f}s",
        f"  Beeps: {analysis.num_beeps}",
        f"  Avg frequency: {analysis.avg_frequency:.0f} Hz",
        f"  Frequency range: {analysis.freq_range[0]:.0f} - {analysis.freq_range[1]:.0f} Hz",
        f"  Pitch direction: {analysis.pitch_direction}",
        f"  Tempo: {analysis.tempo}",
        f"  Complexity: {analysis.complexity}",
    ]

    if analysis.alternative_interpretations:
        lines.append(f"")
        lines.append(f"Alternative interpretations:")
        for alt in analysis.alternative_interpretations:
            lines.append(f"  - \"{alt}\"")

    return "\n".join(lines)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='R2-D2 Translator')
    parser.add_argument('--to-r2d2', '-t', type=str, help='Text to translate to R2D2')
    parser.add_argument('--from-r2d2', '-f', type=str, help='R2D2 WAV file to translate')
    parser.add_argument('--output', '-o', type=str, default='r2d2_translated.wav')
    parser.add_argument('--emotion', '-e', type=str, help='Emotion for to-R2D2')
    parser.add_argument('--play', '-p', action='store_true')

    args = parser.parse_args()

    translator = R2D2Translator()

    if args.to_r2d2:
        print(f"Translating to R2D2: '{args.to_r2d2}'")
        audio = translator.to_r2d2(args.to_r2d2, emotion=args.emotion)
        translator.save(args.output, audio)
        print(f"Saved: {args.output}")

        if args.play:
            translator.play(audio)

    elif args.from_r2d2:
        print(f"Translating from R2D2: {args.from_r2d2}")
        analysis = translator.from_r2d2(args.from_r2d2)
        print()
        print(format_analysis(analysis))

    else:
        # Demo
        print("Demo: Round-trip translation")
        print("=" * 50)

        text = "I found something important!"
        print(f"Original: '{text}'")

        audio = translator.to_r2d2(text, emotion='excited')
        translator.save("demo_r2d2.wav", audio)
        print(f"Generated R2D2 audio: demo_r2d2.wav")

        analysis = translator.from_r2d2(audio)
        print()
        print("Translation back:")
        print(format_analysis(analysis))

        if args.play:
            translator.play(audio)
