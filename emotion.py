"""Personality and emotion tracking with optional persistence."""

from dataclasses import dataclass
from pathlib import Path

from typing import Any


@dataclass
class EmotionState:
    """Track and persist the bot's emotional state."""

    mood_file: str = "mood.txt"
    mood: float = 0.0  # range [-1, 1]

    def __post_init__(self) -> None:
        self._path = Path(self.mood_file)
        if self._path.exists():
            try:
                self.mood = float(self._path.read_text())
            except Exception:
                self.mood = 0.0

    _sentiment: Any | None = None

    def _analyze(self, text: str) -> float:
        """Return sentiment score in range [-1, 1]."""
        if self._sentiment is None:
            from transformers import pipeline

            self._sentiment = pipeline("sentiment-analysis")
        result = self._sentiment(text[:512])[0]
        score = result["score"]
        if result["label"].upper().startswith("NEG"):
            score = -score
        return float(score)

    def update(self, user_text: str) -> None:
        score = self._analyze(user_text)
        self.mood = max(-1.0, min(1.0, self.mood * 0.8 + score * 0.5))
        self._path.write_text(f"{self.mood}")

    def describe(self) -> str:
        if self.mood > 0.5:
            return "happy"
        if self.mood < -0.5:
            return "sad"
        return "neutral"
