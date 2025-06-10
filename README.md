# Simple LLM Example

This project contains a minimal example of loading and fine-tuning a language model using the [HuggingFace Transformers](https://github.com/huggingface/transformers) library.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Generate text with a pre-trained model:

```bash
python main.py
```

Fine-tune the model on custom text data:

```bash
python -c "from llm import LLM; model = LLM(); model.fine_tune('sample_data.txt')"
```

The fine-tuned model will be saved in the `finetuned/` directory.
