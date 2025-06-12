"""Intent understanding and response planning using zero-shot classification."""

from typing import List


class Planner:
    """Classify intent from user text."""

    def __init__(self) -> None:
        self._pipe = None
        self.labels: List[str] = ["greeting", "question", "command", "statement"]

    def _load(self) -> None:
        from transformers import pipeline

        self._pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def analyze(self, user_text: str) -> str:
        if self._pipe is None:
            self._load()
        result = self._pipe(user_text, self.labels)
        label = result["labels"][0]
        return f"{label}: {user_text}"
