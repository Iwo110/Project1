"""Minimal language model wrapper used by the chatbot.

This module provides a small utility class for loading a HuggingFace model,
generating text with configurable decoding parameters and fine tuning on a text
dataset.  It purposefully keeps the interface light weight so other scripts can
use it without needing to know the details of the underlying transformers API.
"""

from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

class LLM:
    """Small helper around a causal language model."""

    def __init__(self, model_name: str = "distilgpt2", device: Optional[str] = None):
        """Initialize the language model on the specified device."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Generate text based on a given prompt using sampling."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def fine_tune(
        self,
        dataset_path: str,
        *,
        output_dir: str = "./finetuned",
        epochs: int = 1,
        batch_size: int = 1,
    ) -> None:
        """Fine tune the model on a plain text dataset."""
        import datasets

        dataset = datasets.load_dataset("text", data_files={"train": dataset_path})
        tokenized = dataset.map(
            lambda e: self.tokenizer(e["text"], truncation=True), batched=True
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized["train"],
            data_collator=data_collator,
        )
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

