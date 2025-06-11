"""Simple command line interface to interact with the chatbot."""

import argparse
from pathlib import Path

from chatbot import ChatBot

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple chatbot")
    parser.add_argument("--model", default="distilgpt2", help="Model name")
    parser.add_argument(
        "--log-dir", default="logs", help="Directory where chat logs are stored"
    )
    parser.add_argument(
        "--memory-file",
        default=None,
        help="Optional file to persist conversation history",
    )
    parser.add_argument("--fine-tune", action="store_true", help="Fine tune on logs and exit")
    args = parser.parse_args()

    bot = ChatBot(model_name=args.model, log_dir=args.log_dir, memory_file=args.memory_file)

    if args.fine_tune:
        bot.fine_tune_on_logs()
        print("Model fine tuned on logs.")
        return

    print("Interactive chat session. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = bot.chat(user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
