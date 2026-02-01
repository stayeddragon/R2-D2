"""
R2-D2 Voice Synthesis Package

Convert text or speech to R2-D2 style beeps and whistles.

Quick Start:
    from voice import R2D2Voice

    voice = R2D2Voice()
    audio = voice.text_to_r2d2("Hello there!")
    voice.save("hello.wav", audio)

With Emotions:
    from voice import R2D2VoiceEnhanced

    voice = R2D2VoiceEnhanced()
    audio = voice.speak("I'm so happy!", emotion='happy')
    voice.save("happy.wav", audio)

From Speech:
    voice = R2D2Voice()
    speech = voice.load("my_voice.wav")
    r2d2 = voice.speech_to_r2d2(speech)
    voice.save("output.wav", r2d2)
"""

from .r2d2_voice import (
    R2D2Voice,
    ProsodyExtractor,
    R2D2Synthesizer,
    TTSEngine,
    text_to_r2d2,
    speech_to_r2d2,
    SAMPLE_RATE,
)

from .r2d2_emotions import (
    EmotionProfile,
    EmotionModulator,
    EMOTIONS,
    detect_emotion_from_text,
    list_emotions,
    get_emotion_description,
)

from .r2d2_enhanced import (
    R2D2VoiceEnhanced,
    EnhancedR2D2Synthesizer,
    speak,
    convert,
    beep,
)

__all__ = [
    # Core
    'R2D2Voice',
    'ProsodyExtractor',
    'R2D2Synthesizer',
    'TTSEngine',
    'text_to_r2d2',
    'speech_to_r2d2',
    'SAMPLE_RATE',
    # Emotions
    'EmotionProfile',
    'EmotionModulator',
    'EMOTIONS',
    'detect_emotion_from_text',
    'list_emotions',
    'get_emotion_description',
    # Enhanced
    'R2D2VoiceEnhanced',
    'EnhancedR2D2Synthesizer',
    'speak',
    'convert',
    'beep',
]

__version__ = '1.0.0'
