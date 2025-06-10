from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import datasets

class LLM:
    def __init__(self, model_name: str = "distilgpt2"):
        """Initialize the language model with the given model name."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text based on a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def fine_tune(self, dataset_path: str, output_dir: str = "./finetuned", epochs: int = 1):
        """Fine tune the model on the text dataset located at dataset_path."""
        dataset = datasets.load_dataset("text", data_files={"train": dataset_path})
        tokenized = dataset.map(lambda e: self.tokenizer(e["text"], truncation=True), batched=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs, per_device_train_batch_size=1)
        trainer = Trainer(model=self.model, args=args, train_dataset=tokenized["train"], data_collator=data_collator)
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
