"""Effector layer for producing outputs."""

from dataclasses import dataclass

@dataclass
class TextOutput:
    def speak(self, text: str) -> None:
        print(f"Bot: {text}")

# Placeholder for future text-to-speech and avatar animation
class SpeechSynthesizer:
    def say(self, text: str) -> None:
        raise NotImplementedError

class Avatar:
    def animate(self, emotion: str) -> None:
        raise NotImplementedError
