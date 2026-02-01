"""
R2-D2 Emotion and Personality System

Defines how different emotions affect the R2D2 voice synthesis.
Emotions influence:
- Frequency range and center
- Speed/tempo
- Amount of frequency modulation (wobble)
- Chirpiness (frequency sweeps)
- Noise blend
- Beep patterns

Based on analysis of R2-D2 sounds from the Star Wars films:
- Happy/Excited: Higher frequencies, faster, more chirps, upward sweeps
- Sad/Worried: Lower frequencies, slower, downward sweeps, more tremolo
- Angry/Urgent: Harsh, rapid, staccato, higher noise content
- Curious: Questioning upward inflections at end
- Affirmative: Short, confident beeps
- Negative: Descending, buzzy tones
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class EmotionProfile:
    """Defines synthesis parameters for an emotion."""

    # Frequency settings
    freq_center: float = 800      # Center frequency in Hz
    freq_range: float = 1.0       # Multiplier for frequency range
    freq_shift: float = 0         # Shift all frequencies up/down (Hz)

    # Timing
    tempo: float = 1.0            # Speed multiplier (1.0 = normal)
    attack: float = 0.005         # Attack time in seconds
    release: float = 0.01         # Release time in seconds

    # Character
    expressiveness: float = 1.5   # Pitch movement exaggeration
    chirpiness: float = 0.3       # Probability of frequency sweeps
    chirp_direction: float = 0    # -1 = down, 0 = random, 1 = up
    noise_blend: float = 0.1      # Noise amount
    wobble_rate: float = 20       # Vibrato rate in Hz
    wobble_depth: float = 30      # Vibrato depth in Hz

    # Waveform
    square_mix: float = 0.3       # Amount of squared wave (0-1)
    harmonic2: float = 0.15       # Second harmonic amount
    harmonic3: float = 0.08       # Third harmonic amount

    # Special effects
    staccato: float = 0           # Amount of staccato (0-1)
    tremolo_rate: float = 0       # Amplitude tremolo rate (0 = off)
    tremolo_depth: float = 0      # Amplitude tremolo depth


# Predefined emotion profiles
EMOTIONS: Dict[str, EmotionProfile] = {
    # === POSITIVE EMOTIONS ===

    'happy': EmotionProfile(
        freq_center=1000,
        freq_range=1.2,
        freq_shift=100,
        tempo=1.1,
        expressiveness=1.8,
        chirpiness=0.5,
        chirp_direction=0.7,  # Mostly upward
        wobble_rate=25,
        wobble_depth=40,
        harmonic2=0.2,
    ),

    'excited': EmotionProfile(
        freq_center=1200,
        freq_range=1.5,
        freq_shift=200,
        tempo=1.3,
        attack=0.003,
        expressiveness=2.0,
        chirpiness=0.7,
        chirp_direction=0.8,
        noise_blend=0.15,
        wobble_rate=30,
        wobble_depth=50,
    ),

    'playful': EmotionProfile(
        freq_center=1100,
        freq_range=1.3,
        tempo=1.15,
        expressiveness=2.0,
        chirpiness=0.6,
        chirp_direction=0,  # Random
        wobble_rate=35,
        wobble_depth=60,
        staccato=0.3,
    ),

    'affirmative': EmotionProfile(
        freq_center=900,
        freq_range=0.8,
        freq_shift=50,
        tempo=1.0,
        attack=0.003,
        release=0.015,
        expressiveness=1.2,
        chirpiness=0.2,
        chirp_direction=0.5,
        square_mix=0.25,
    ),

    'proud': EmotionProfile(
        freq_center=950,
        freq_range=1.0,
        freq_shift=100,
        tempo=0.9,
        expressiveness=1.5,
        chirpiness=0.3,
        chirp_direction=0.4,
        harmonic2=0.2,
        harmonic3=0.1,
    ),

    # === NEGATIVE EMOTIONS ===

    'sad': EmotionProfile(
        freq_center=600,
        freq_range=0.7,
        freq_shift=-100,
        tempo=0.7,
        attack=0.01,
        release=0.02,
        expressiveness=1.0,
        chirpiness=0.2,
        chirp_direction=-0.8,  # Downward
        wobble_rate=8,
        wobble_depth=20,
        tremolo_rate=5,
        tremolo_depth=0.2,
    ),

    'worried': EmotionProfile(
        freq_center=700,
        freq_range=0.9,
        freq_shift=-50,
        tempo=0.85,
        expressiveness=1.3,
        chirpiness=0.4,
        chirp_direction=-0.5,
        wobble_rate=15,
        wobble_depth=35,
        tremolo_rate=8,
        tremolo_depth=0.15,
    ),

    'scared': EmotionProfile(
        freq_center=900,
        freq_range=1.4,
        tempo=1.2,
        attack=0.002,
        expressiveness=2.0,
        chirpiness=0.6,
        chirp_direction=0,
        noise_blend=0.2,
        wobble_rate=40,
        wobble_depth=80,
        tremolo_rate=12,
        tremolo_depth=0.3,
    ),

    'negative': EmotionProfile(
        freq_center=500,
        freq_range=0.6,
        freq_shift=-150,
        tempo=0.8,
        expressiveness=0.8,
        chirpiness=0.3,
        chirp_direction=-1.0,  # All downward
        noise_blend=0.2,
        square_mix=0.5,
    ),

    'annoyed': EmotionProfile(
        freq_center=700,
        freq_range=0.8,
        tempo=1.1,
        attack=0.002,
        release=0.005,
        expressiveness=1.5,
        chirpiness=0.4,
        noise_blend=0.25,
        square_mix=0.4,
        staccato=0.4,
    ),

    'angry': EmotionProfile(
        freq_center=600,
        freq_range=1.0,
        tempo=1.2,
        attack=0.001,
        release=0.003,
        expressiveness=1.8,
        chirpiness=0.5,
        chirp_direction=-0.3,
        noise_blend=0.35,
        square_mix=0.6,
        staccato=0.5,
        harmonic3=0.15,
    ),

    # === QUESTIONING/CURIOUS ===

    'curious': EmotionProfile(
        freq_center=900,
        freq_range=1.2,
        tempo=0.95,
        expressiveness=1.6,
        chirpiness=0.5,
        chirp_direction=0.9,  # End with upward
        wobble_rate=18,
        wobble_depth=25,
    ),

    'questioning': EmotionProfile(
        freq_center=850,
        freq_range=1.3,
        tempo=0.9,
        expressiveness=1.8,
        chirpiness=0.6,
        chirp_direction=1.0,  # Strong upward at end
        wobble_rate=15,
        wobble_depth=30,
    ),

    'confused': EmotionProfile(
        freq_center=800,
        freq_range=1.1,
        tempo=0.85,
        expressiveness=1.4,
        chirpiness=0.5,
        chirp_direction=0,  # Random
        wobble_rate=20,
        wobble_depth=40,
        tremolo_rate=6,
        tremolo_depth=0.1,
    ),

    # === URGENT/ALERT ===

    'urgent': EmotionProfile(
        freq_center=1000,
        freq_range=1.3,
        tempo=1.4,
        attack=0.001,
        release=0.002,
        expressiveness=2.0,
        chirpiness=0.4,
        noise_blend=0.2,
        square_mix=0.4,
        staccato=0.6,
    ),

    'alert': EmotionProfile(
        freq_center=1100,
        freq_range=1.2,
        freq_shift=100,
        tempo=1.3,
        attack=0.002,
        expressiveness=1.8,
        chirpiness=0.5,
        chirp_direction=0.6,
        noise_blend=0.15,
        staccato=0.4,
    ),

    'warning': EmotionProfile(
        freq_center=800,
        freq_range=0.8,
        tempo=1.0,
        expressiveness=1.2,
        chirpiness=0.3,
        noise_blend=0.25,
        square_mix=0.5,
        tremolo_rate=10,
        tremolo_depth=0.4,
    ),

    # === NEUTRAL ===

    'neutral': EmotionProfile(
        # All defaults
    ),

    'calm': EmotionProfile(
        freq_center=750,
        freq_range=0.9,
        tempo=0.9,
        attack=0.008,
        release=0.015,
        expressiveness=1.2,
        chirpiness=0.2,
        wobble_rate=12,
        wobble_depth=15,
    ),

    'thoughtful': EmotionProfile(
        freq_center=700,
        freq_range=0.85,
        tempo=0.8,
        expressiveness=1.3,
        chirpiness=0.35,
        wobble_rate=10,
        wobble_depth=20,
    ),
}


class EmotionModulator:
    """
    Modulates R2D2 synthesis based on emotion.

    Can blend between emotions for smooth transitions.
    """

    def __init__(self, default_emotion: str = 'neutral'):
        self.current_emotion = default_emotion
        self.target_emotion = default_emotion
        self.blend_factor = 1.0

    def set_emotion(self, emotion: str):
        """Set the current emotion."""
        if emotion not in EMOTIONS:
            raise ValueError(f"Unknown emotion: {emotion}. "
                           f"Available: {list(EMOTIONS.keys())}")
        self.current_emotion = emotion
        self.target_emotion = emotion
        self.blend_factor = 1.0

    def transition_to(self, emotion: str, blend: float = 0.5):
        """
        Start transitioning to a new emotion.

        Args:
            emotion: Target emotion
            blend: How much of the new emotion (0-1)
        """
        if emotion not in EMOTIONS:
            raise ValueError(f"Unknown emotion: {emotion}")
        self.target_emotion = emotion
        self.blend_factor = blend

    def get_profile(self) -> EmotionProfile:
        """Get the current (possibly blended) emotion profile."""
        if self.current_emotion == self.target_emotion or self.blend_factor >= 1.0:
            return EMOTIONS[self.current_emotion]

        # Blend between emotions
        current = EMOTIONS[self.current_emotion]
        target = EMOTIONS[self.target_emotion]

        return self._blend_profiles(current, target, self.blend_factor)

    def _blend_profiles(
        self,
        a: EmotionProfile,
        b: EmotionProfile,
        t: float
    ) -> EmotionProfile:
        """Linearly interpolate between two emotion profiles."""
        def lerp(x, y):
            return x + (y - x) * t

        return EmotionProfile(
            freq_center=lerp(a.freq_center, b.freq_center),
            freq_range=lerp(a.freq_range, b.freq_range),
            freq_shift=lerp(a.freq_shift, b.freq_shift),
            tempo=lerp(a.tempo, b.tempo),
            attack=lerp(a.attack, b.attack),
            release=lerp(a.release, b.release),
            expressiveness=lerp(a.expressiveness, b.expressiveness),
            chirpiness=lerp(a.chirpiness, b.chirpiness),
            chirp_direction=lerp(a.chirp_direction, b.chirp_direction),
            noise_blend=lerp(a.noise_blend, b.noise_blend),
            wobble_rate=lerp(a.wobble_rate, b.wobble_rate),
            wobble_depth=lerp(a.wobble_depth, b.wobble_depth),
            square_mix=lerp(a.square_mix, b.square_mix),
            harmonic2=lerp(a.harmonic2, b.harmonic2),
            harmonic3=lerp(a.harmonic3, b.harmonic3),
            staccato=lerp(a.staccato, b.staccato),
            tremolo_rate=lerp(a.tremolo_rate, b.tremolo_rate),
            tremolo_depth=lerp(a.tremolo_depth, b.tremolo_depth),
        )


def detect_emotion_from_text(text: str) -> str:
    """
    Simple rule-based emotion detection from text.

    For production, you'd want to use a proper sentiment analysis model.
    """
    text = text.lower()

    # Question detection
    if '?' in text or any(w in text for w in ['what', 'why', 'how', 'where', 'when', 'who']):
        if any(w in text for w in ['worried', 'scared', 'afraid']):
            return 'scared'
        return 'questioning'

    # Exclamation detection
    if '!' in text:
        if any(w in text for w in ['help', 'danger', 'warning', 'alert', 'emergency']):
            return 'urgent'
        if any(w in text for w in ['great', 'awesome', 'wonderful', 'amazing', 'yes']):
            return 'excited'
        if any(w in text for w in ['no', 'stop', 'bad', 'wrong']):
            return 'angry'

    # Positive words
    if any(w in text for w in ['happy', 'glad', 'joy', 'love', 'like', 'good', 'great', 'nice']):
        return 'happy'

    if any(w in text for w in ['yes', 'okay', 'sure', 'right', 'correct', 'agree']):
        return 'affirmative'

    # Negative words
    if any(w in text for w in ['no', 'not', 'never', 'wrong', 'bad', 'disagree']):
        return 'negative'

    if any(w in text for w in ['sad', 'sorry', 'miss', 'unfortunately']):
        return 'sad'

    if any(w in text for w in ['worried', 'concern', 'anxious', 'nervous']):
        return 'worried'

    if any(w in text for w in ['angry', 'mad', 'furious', 'hate']):
        return 'angry'

    if any(w in text for w in ['confused', 'uncertain', 'unsure', "don't know"]):
        return 'confused'

    if any(w in text for w in ['curious', 'wonder', 'interesting']):
        return 'curious'

    if any(w in text for w in ['careful', 'warning', 'caution', 'danger']):
        return 'warning'

    return 'neutral'


def list_emotions() -> list:
    """Return list of available emotions."""
    return list(EMOTIONS.keys())


def get_emotion_description(emotion: str) -> str:
    """Get a description of what an emotion sounds like."""
    descriptions = {
        'happy': 'High-pitched, bouncy beeps with upward chirps',
        'excited': 'Fast, high, energetic with lots of sweeps',
        'playful': 'Varied, bouncy, with random pitch changes',
        'affirmative': 'Short, confident, slightly upward beep',
        'proud': 'Warm, full-bodied tones with harmonics',
        'sad': 'Low, slow, descending tones with tremolo',
        'worried': 'Wavering, uncertain, slightly trembling',
        'scared': 'Rapid, shaky, high-pitched with noise',
        'negative': 'Low, buzzy, descending "raspberry" sound',
        'annoyed': 'Short, sharp, slightly harsh beeps',
        'angry': 'Harsh, staccato, noisy buzzing',
        'curious': 'Rising inflection at the end, questioning',
        'questioning': 'Clear upward sweep at the end',
        'confused': 'Wavering, uncertain pitch movements',
        'urgent': 'Fast, staccato, attention-grabbing',
        'alert': 'Sharp, clear, high-pitched warning',
        'warning': 'Steady, pulsing, cautionary tone',
        'neutral': 'Balanced, natural R2D2 beeping',
        'calm': 'Slow, gentle, soothing beeps',
        'thoughtful': 'Slow, measured, contemplative tones',
    }
    return descriptions.get(emotion, 'Unknown emotion')
