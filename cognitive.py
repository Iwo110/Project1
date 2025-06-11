"""Cognitive layer with language model and memory."""

from dataclasses import dataclass
from typing import List

from llm import LLM
from memory import Memory
from vector_memory import VectorMemory


@dataclass
class CognitiveCore:
    model_name: str = "distilgpt2"
    memory_file: str | None = None
    vector_memory_path: str | None = None

    def __post_init__(self) -> None:
        self.llm = LLM(self.model_name)
        self.memory = Memory(self.memory_file) if self.memory_file else None
        self.vector_memory = (
            VectorMemory(self.vector_memory_path)
            if self.vector_memory_path
            else None
        )

    def generate(self, prompt: str) -> str:
        context = ""
        if self.vector_memory:
            retrieved = self.vector_memory.search(prompt, top_k=3)
            if retrieved:
                context = "\n".join(retrieved) + "\n"
        return self.llm.generate(context + prompt)

    def remember(self, lines: List[str]) -> None:
        if self.memory:
            for line in lines:
                self.memory.append(line)
        if self.vector_memory:
            for line in lines:
                self.vector_memory.add(line)
