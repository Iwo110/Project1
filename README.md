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
