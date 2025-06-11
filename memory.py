"""Persistent conversation memory for the chatbot."""

from pathlib import Path
from typing import Iterable, List


class Memory:
    """Lightweight helper to store and load chat history."""

    def __init__(self, path: str = "memory.txt") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[str]:
        """Return all stored lines as a list of strings."""
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def append(self, line: str) -> None:
        """Append a single line to memory."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

    def overwrite(self, lines: Iterable[str]) -> None:
        """Replace memory with the provided lines."""
        with open(self.path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line.rstrip() + "\n")

    def clear(self) -> None:
        """Delete the memory file."""
        if self.path.exists():
            self.path.unlink()
