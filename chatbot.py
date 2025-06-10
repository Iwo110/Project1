from llm import LLM
import os

class ChatBot:
    """Simple chatbot that keeps conversation history and can fine-tune on logs."""

    def __init__(self, model_name: str = "distilgpt2", log_path: str = "chat_logs.txt"):
        self.llm = LLM(model_name)
        self.history = []
        self.log_path = log_path

    def chat(self, user_input: str) -> str:
        """Generate a response to the user input and update history."""
        self.history.append(f"User: {user_input}")
        prompt = "\n".join(self.history) + "\nAssistant:"
        response = self.llm.generate(prompt)
        # Keep only the text after the last 'Assistant:' tag if present
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        self.history.append(f"Assistant: {response}")
        with open(self.log_path, "a") as f:
            f.write(f"User: {user_input}\nAssistant: {response}\n")
        return response

    def fine_tune_on_logs(self):
        """Fine-tune the underlying model on the logged conversations."""
        if os.path.exists(self.log_path):
            self.llm.fine_tune(self.log_path)
