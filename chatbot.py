"""Chatbot module wrapping the LLM with conversation management utilities."""

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

from llm import LLM

class ChatBot:
    """Simple chatbot that keeps conversation history and can fine-tune on logs."""

    def __init__(
        self,
        model_name: str = "distilgpt2",
        *,
        log_dir: str = "logs",
        max_history: int = 10,
    ) -> None:
        self.llm = LLM(model_name)
        self.history: list[str] = []
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"chat_{timestamp}.txt"
        self.max_history = max_history

    def chat(self, user_input: str) -> str:
        """Generate a response to the user input and update history."""
        self.history.append(f"User: {user_input}")
        # truncate history so the prompt does not grow unbounded
        if len(self.history) > 2 * self.max_history:
            self.history = self.history[-2 * self.max_history :]

        prompt = "\n".join(self.history) + "\nAssistant:"
        response = self.llm.generate(prompt)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        self.history.append(f"Assistant: {response}")

        with open(self.log_path, "a") as f:
            f.write(f"User: {user_input}\nAssistant: {response}\n")

        return response

    def fine_tune_on_logs(self):
        """Fine-tune the underlying model on the logged conversations."""
        for log_file in self.log_dir.glob("chat_*.txt"):
            self.llm.fine_tune(str(log_file))

    def load_history(self, lines: Iterable[str]) -> None:
        """Load conversation history from an iterable of lines."""
        self.history = [line.strip() for line in lines if line.strip()]

    def reset_history(self) -> None:
        """Clear the in-memory history and delete the current log file."""
        self.history = []
        if self.log_path.exists():
            self.log_path.unlink()

