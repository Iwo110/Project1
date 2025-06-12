"""Effector layer for producing outputs.

Besides printing text, this module can optionally use ``pyttsx3`` to vocalize
responses so the bot is able to speak aloud. Avatar animation remains a stub.
"""

from dataclasses import dataclass

@dataclass
class TextOutput:
    def speak(self, text: str) -> None:
        print(f"Bot: {text}")

# Placeholder for future text-to-speech and avatar animation
class SpeechSynthesizer:
    """Simple text-to-speech implementation using ``pyttsx3``."""

    def __init__(self) -> None:
        import pyttsx3

        self.engine = pyttsx3.init()

    def say(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()

class Avatar:
    def animate(self, emotion: str) -> None:
        raise NotImplementedError
