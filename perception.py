"""Perception layer components.

This module collects input from various modalities. For now only text
is implemented. Future extensions will add speech recognition and image
analysis.
"""

from dataclasses import dataclass

@dataclass
class TextInput:
    """Simple text input interface."""

    def receive(self) -> str:
        return input("You: ")

# Placeholder classes for future speech and vision modules
class SpeechRecognizer:
    def transcribe(self, audio_bytes: bytes) -> str:
        raise NotImplementedError

class VisionAnalyzer:
    def describe(self, image_bytes: bytes) -> str:
        raise NotImplementedError
