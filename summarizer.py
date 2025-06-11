from __future__ import annotations

from typing import List

from transformers import pipeline


class Summarizer:
    """Utility class for summarizing text passages."""

    def __init__(self, model_name: str = "google-t5/t5-small") -> None:
        self.pipe = pipeline("summarization", model=model_name)

    def summarize(self, lines: List[str], max_length: int = 60) -> str:
        """Return a summary of the provided lines."""
        text = "\n".join(lines)
        result = self.pipe(text, max_length=max_length, min_length=10, do_sample=False)
        return result[0]["summary_text"]
