"""Perception layer components.

This module collects input from various modalities. Besides plain text it now
offers a simple speech recognizer based on the ``speech_recognition`` package.
Image analysis remains a placeholder.
"""

from dataclasses import dataclass

@dataclass
class TextInput:
    """Simple text input interface."""

    def receive(self) -> str:
        return input("You: ")

# Placeholder classes for future speech and vision modules
class SpeechRecognizer:
    """Convert microphone input to text using ``speech_recognition``."""

    def __init__(self) -> None:
        import speech_recognition as sr

        self.sr = sr.Recognizer()
        self.mic = sr.Microphone()

    def listen(self, timeout: float | None = None) -> str:
        import speech_recognition as sr

        with self.mic as source:
            audio = self.sr.listen(source, timeout=timeout)
        try:
            return self.sr.recognize_google(audio, language="pl-PL")
        except sr.UnknownValueError:
            return ""

    def transcribe(self, audio_bytes: bytes) -> str:  # legacy API
        with open("_temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        with self.mic as source:
            audio = self.sr.record(source, duration=0)  # dummy to init
        with sr.AudioFile("_temp_audio.wav") as source:
            audio = self.sr.record(source)
        try:
            return self.sr.recognize_google(audio, language="pl-PL")
        except sr.UnknownValueError:
            return ""

class VisionAnalyzer:
    """Generate captions for images using a pre-trained transformer."""

    def __init__(self, model_name: str = "nlpconnect/vit-gpt2-image-captioning") -> None:
        self.model_name = model_name
        self.pipe = None

    def describe(self, image_path: str) -> str:
        """Return a textual description of the provided image file."""
        if self.pipe is None:
            from transformers import pipeline

            self.pipe = pipeline("image-to-text", model=self.model_name)
        result = self.pipe(image_path)[0]["generated_text"]
        return result
