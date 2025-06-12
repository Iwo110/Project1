from __future__ import annotations

from typing import Any

from transformers import pipeline


class SafetyChecker:
    """Detect toxic content in text responses."""

    def __init__(self, model_name: str = "unitary/toxic-bert") -> None:
        self.model_name = model_name
        self.pipe: Any | None = None

    def _load(self) -> None:
        self.pipe = pipeline("text-classification", model=self.model_name)

    def is_toxic(self, text: str, threshold: float = 0.5) -> bool:
        """Return True if the text is likely toxic."""
        if self.pipe is None:
            self._load()
        result = self.pipe(text[:512])[0]
        label = result["label"].lower()
        score = float(result["score"])
        return label == "toxic" and score >= threshold
