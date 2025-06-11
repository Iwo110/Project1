"""Personality and emotion tracking."""

class EmotionState:
    def __init__(self) -> None:
        self.mood = 0.0  # range [-1, 1]

    def update(self, user_text: str) -> None:
        delta = (len(user_text) % 5 - 2) * 0.1
        self.mood = max(-1.0, min(1.0, self.mood * 0.9 + delta))

    def describe(self) -> str:
        if self.mood > 0.5:
            return "happy"
        if self.mood < -0.5:
            return "sad"
        return "neutral"
