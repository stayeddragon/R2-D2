#!/usr/bin/env python3
"""
R2-D2 Voice Examples

Run this script to generate sample R2D2 audio files demonstrating
various features of the voice synthesis system.

Usage:
    python examples.py
    python examples.py --play  # Also play the audio

This will create several WAV files in the current directory.
"""

import os
import sys
import argparse

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from r2d2_voice import R2D2Voice, text_to_r2d2
from r2d2_enhanced import R2D2VoiceEnhanced, speak, beep
from r2d2_emotions import list_emotions, get_emotion_description


def example_basic():
    """Basic text-to-R2D2 conversion."""
    print("\n=== Basic Text to R2D2 ===")

    voice = R2D2Voice()

    # Simple greeting
    audio = voice.text_to_r2d2("Hello there!")
    voice.save("example_hello.wav", audio)
    print("Created: example_hello.wav")

    return audio


def example_emotions():
    """Demonstrate different emotions."""
    print("\n=== Emotion Examples ===")

    voice = R2D2VoiceEnhanced()
    samples = []

    emotion_texts = [
        ('happy', "I'm so happy to see you!"),
        ('sad', "I miss my friends..."),
        ('excited', "This is amazing!"),
        ('angry', "I don't like that at all!"),
        ('curious', "What's that over there?"),
        ('scared', "Oh no, what was that sound?"),
        ('urgent', "Warning! Danger detected!"),
        ('affirmative', "Yes, I understand."),
        ('negative', "No, that's not right."),
    ]

    for emotion, text in emotion_texts:
        audio = voice.speak(text, emotion=emotion)
        filename = f"example_{emotion}.wav"
        voice.save(filename, audio)
        print(f"Created: {filename} - '{text}'")
        samples.append((emotion, audio))

    return samples


def example_auto_emotion():
    """Show automatic emotion detection."""
    print("\n=== Auto Emotion Detection ===")

    voice = R2D2VoiceEnhanced(auto_detect_emotion=True)

    texts = [
        "What is that thing?",
        "Yay! We did it!",
        "Oh no, this is terrible...",
        "HELP! Emergency!",
        "Yes, I agree with you.",
        "No, I don't think so.",
    ]

    from r2d2_emotions import detect_emotion_from_text

    for text in texts:
        detected = detect_emotion_from_text(text)
        audio = voice.speak(text)  # Auto-detects
        filename = f"example_auto_{detected}.wav"
        voice.save(filename, audio)
        print(f"'{text}' -> detected: {detected}")


def example_beeps():
    """Generate various beep patterns."""
    print("\n=== Beep Patterns ===")

    voice = R2D2VoiceEnhanced()

    # Single beeps with different emotions
    for emotion in ['happy', 'sad', 'curious', 'urgent']:
        audio = voice.generate_beep(emotion=emotion)
        filename = f"example_beep_{emotion}.wav"
        voice.save(filename, audio)
        print(f"Created: {filename}")

    # Sequences
    patterns = ['ascending', 'descending', 'question', 'exclaim', 'random']
    for pattern in patterns:
        audio = voice.generate_sequence(pattern=pattern, length=5)
        filename = f"example_sequence_{pattern}.wav"
        voice.save(filename, audio)
        print(f"Created: {filename}")


def example_expressiveness():
    """Show different expressiveness levels."""
    print("\n=== Expressiveness Levels ===")

    from r2d2_voice import R2D2Voice

    text = "I have something important to tell you."

    for expr in [0.5, 1.0, 1.5, 2.0, 2.5]:
        voice = R2D2Voice(expressiveness=expr)
        audio = voice.text_to_r2d2(text)
        filename = f"example_expr_{expr:.1f}.wav"
        voice.save(filename, audio)
        print(f"Created: {filename} (expressiveness={expr})")


def example_conversation():
    """Simulate a back-and-forth conversation."""
    print("\n=== Conversation Demo ===")

    voice = R2D2VoiceEnhanced()

    # C-3PO asks, R2D2 responds (we generate R2's responses)
    exchanges = [
        ("curious", "What did you find?"),
        ("excited", "A secret message!"),
        ("questioning", "From Princess Leia?"),
        ("affirmative", "Yes! We must deliver it!"),
        ("urgent", "The Empire is coming!"),
        ("scared", "They're getting closer!"),
        ("alert", "We need to escape now!"),
    ]

    all_audio = []
    import numpy as np

    for emotion, meaning in exchanges:
        audio = voice.speak(meaning, emotion=emotion)
        all_audio.append(audio)
        # Add small pause
        all_audio.append(np.zeros(int(0.3 * 22050), dtype=np.float32))

    # Concatenate all
    conversation = np.concatenate(all_audio)
    voice.save("example_conversation.wav", conversation)
    print("Created: example_conversation.wav")

    for emotion, meaning in exchanges:
        print(f"  R2D2 ({emotion}): \"{meaning}\"")


def example_list_emotions():
    """Print all available emotions."""
    print("\n=== Available Emotions ===")

    for emotion in list_emotions():
        desc = get_emotion_description(emotion)
        print(f"  {emotion:15s} - {desc}")


def main():
    parser = argparse.ArgumentParser(description='R2-D2 Voice Examples')
    parser.add_argument('--play', '-p', action='store_true',
                        help='Play the last generated audio')
    parser.add_argument('--example', '-e', type=str,
                        choices=['basic', 'emotions', 'auto', 'beeps',
                                'expressiveness', 'conversation', 'all'],
                        default='all',
                        help='Which example to run')
    args = parser.parse_args()

    print("=" * 50)
    print("R2-D2 Voice Synthesis Examples")
    print("=" * 50)

    example_list_emotions()

    last_audio = None
    voice = None

    if args.example in ['basic', 'all']:
        last_audio = example_basic()
        voice = R2D2Voice()

    if args.example in ['emotions', 'all']:
        samples = example_emotions()
        if samples:
            last_audio = samples[0][1]
        voice = R2D2VoiceEnhanced()

    if args.example in ['auto', 'all']:
        example_auto_emotion()
        voice = R2D2VoiceEnhanced()

    if args.example in ['beeps', 'all']:
        example_beeps()
        voice = R2D2VoiceEnhanced()

    if args.example in ['expressiveness', 'all']:
        example_expressiveness()
        voice = R2D2Voice()

    if args.example in ['conversation', 'all']:
        example_conversation()
        voice = R2D2VoiceEnhanced()

    print("\n" + "=" * 50)
    print("Examples complete! Check the generated WAV files.")
    print("=" * 50)

    if args.play and last_audio is not None and voice is not None:
        print("\nPlaying last generated audio...")
        voice.play(last_audio)


if __name__ == '__main__':
    main()
