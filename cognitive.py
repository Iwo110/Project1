"""Cognitive layer with language model and memory."""

from dataclasses import dataclass
from typing import List

from llm import LLM
from memory import Memory


@dataclass
class CognitiveCore:
    model_name: str = "distilgpt2"
    memory_file: str | None = None

    def __post_init__(self) -> None:
        self.llm = LLM(self.model_name)
        self.memory = Memory(self.memory_file) if self.memory_file else None

    def generate(self, prompt: str) -> str:
        return self.llm.generate(prompt)

    def remember(self, lines: List[str]) -> None:
        if self.memory:
            for line in lines:
                self.memory.append(line)
