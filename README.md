# Simple Chatbot Example

This project demonstrates a small conversational bot built on top of the
[HuggingFace Transformers](https://github.com/huggingface/transformers) library.
The bot keeps a history of the conversation in timestamped log files and can be
fine-tuned later on those logs.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Start an interactive chat session using the default DistilGPT2 model:

```bash
python main.py
```

To fine-tune the model on the conversation logs after chatting, run:

```bash
python -c "from chatbot import ChatBot; bot = ChatBot(); bot.fine_tune_on_logs()"
```

Fine-tuned models are saved in the `finetuned/` directory.

### Command line options

* `--model` - specify a different model name from HuggingFace.
* `--log-dir` - directory where chat logs will be written.
* `--fine-tune` - fine tune the model on all logs then exit.
* `--memory-file` - optional file used to persist chat history across sessions.

## Evolving Brain

The `brain.py` module builds on the basic `ChatBot` by adding a minimal "mood"
state and an evolution routine. Mood shifts slightly based on the text length of
user messages, and the brain can periodically fine‑tune the underlying model on
the accumulated logs.

The brain persists its mood to `mood.txt` so emotional state is remembered
between sessions. When a memory file is provided, past conversation lines are
loaded at startup and new exchanges are appended automatically.

Example usage:

```python
from brain import Brain
brain = Brain()
while True:
    user = input("You: ")
    if user == "exit":
        break
    reply = brain.talk(user)
    print("Bot:", reply)
    brain.evolve(every_n=5)
```

