# Simple Chatbot Example

This project demonstrates a very small conversational bot built on top of the [HuggingFace Transformers](https://github.com/huggingface/transformers) library. The bot keeps a history of the conversation and can be fine-tuned later on the stored chat logs.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Start an interactive chat session:

```bash
python main.py
```

To fine-tune the model on the conversation logs after chatting, run:

```bash
python -c "from chatbot import ChatBot; bot = ChatBot(); bot.fine_tune_on_logs()"
```

Fine-tuned models are saved in the `finetuned/` directory.
