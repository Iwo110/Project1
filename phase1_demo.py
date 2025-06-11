"""Demonstration of the Phase 1 architecture skeleton."""

from perception import TextInput
from cognitive import CognitiveCore
from emotion import EmotionState
from planning import Planner
from effector import TextOutput
from summarizer import Summarizer


def main() -> None:
    input_module = TextInput()
    core = CognitiveCore(vector_memory_path="demo_memory")
    emotion = EmotionState()
    planner = Planner()
    output = TextOutput()
    summarizer = Summarizer()
    history: list[str] = []

    print("Phase 1 demo. Type 'exit' to quit.")
    while True:
        text = input_module.receive()
        if text.lower() == "exit":
            break
        if text.lower() == "summary":
            summary = summarizer.summarize(history)
            output.speak(f"Summary: {summary}")
            continue
        emotion.update(text)
        intent = planner.analyze(text)
        response = core.generate(f"User: {intent}\nAssistant:")
        history.extend([f"User: {text}", f"Assistant: {response}"])
        core.remember([f"User: {text}", f"Assistant: {response}"])
        output.speak(response)
        print(f"(Emotion: {emotion.describe()})")


if __name__ == "__main__":
    main()
