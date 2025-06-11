"""Simple brain module for evolving the chatbot.

This module introduces a :class:`Brain` that wraps the :class:`ChatBot`
with a minimal notion of "mood" which changes slightly based on the
conversation. The brain can also automatically fine tune the underlying
language model on the accumulated logs to gradually adapt to new data.
The implementation is intentionally lightweight for demonstration.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from chatbot import ChatBot


class Brain:
    """Wrapper around :class:`ChatBot` with simple mood tracking."""

    def __init__(
        self,
        model_name: str = "distilgpt2",
        *,
        log_dir: str = "logs",
        max_history: int = 10,
    ) -> None:
        self.chatbot = ChatBot(model_name=model_name, log_dir=log_dir, max_history=max_history)
        # mood is a number in [-1, 1]; positive means happy, negative sad
        self.mood: float = 0.0
        self.interactions = 0

    def _update_mood(self, text: str) -> None:
        """Naively update mood based on user text length."""
        delta = (len(text) % 5 - 2) * 0.1
        # move mood slightly toward delta but keep in [-1, 1]
        self.mood = max(-1.0, min(1.0, self.mood * 0.9 + delta))

    def talk(self, user_input: str) -> str:
        """Respond to the user and evolve the brain state."""
        self._update_mood(user_input)
        response = self.chatbot.chat(user_input)
        self.interactions += 1
        return response

    def evolve(self, *, every_n: int = 5) -> None:
        """Fine tune the model after a number of interactions."""
        if self.interactions >= every_n:
            self.chatbot.fine_tune_on_logs()
            self.interactions = 0

    def load_history(self, lines: Iterable[str]) -> None:
        self.chatbot.load_history(lines)

    def reset(self) -> None:
        self.chatbot.reset_history()
        self.mood = 0.0
        self.interactions = 0
