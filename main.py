from llm import LLM

if __name__ == "__main__":
    model = LLM()
    prompt = "Once upon a time"
    print("Generating text...")
    print(model.generate(prompt))
