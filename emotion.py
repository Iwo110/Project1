"""Personality and emotion tracking with optional persistence."""

from dataclasses import dataclass
from pathlib import Path


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

    def update(self, user_text: str) -> None:
        delta = (len(user_text) % 5 - 2) * 0.1
        self.mood = max(-1.0, min(1.0, self.mood * 0.9 + delta))
        self._path.write_text(f"{self.mood}")

    def describe(self) -> str:
        if self.mood > 0.5:
            return "happy"
        if self.mood < -0.5:
            return "sad"
        return "neutral"
