"""Demonstration of the Phase 1 architecture skeleton."""

from perception import TextInput, VisionAnalyzer
from cognitive import CognitiveCore
from emotion import EmotionState
from planning import Planner
from effector import TextOutput
from summarizer import Summarizer


def main() -> None:
    input_module = TextInput()
    vision = VisionAnalyzer()
    core = CognitiveCore(
        vector_memory_path="demo_memory", memory_file="demo_history.txt"
    )
    emotion = EmotionState(mood_file="demo_mood.txt")
    planner = Planner()
    output = TextOutput()
    summarizer: Summarizer | None = None
    history: list[str] = []
    if core.memory:
        history.extend(core.memory.load())
        if core.vector_memory and not core.vector_memory.index_path.exists():
            for line in history:
                core.vector_memory.add(line)

    print("Phase 1 demo. Type 'exit' to quit.")
    while True:
        text = input_module.receive()
        if text.lower() == "exit":
            break
        if text.startswith("look "):
            img_path = text.split(" ", 1)[1]
            try:
                description = vision.describe(img_path)
                print(f"Image description: {description}")
                emotion.update(description)
                response = core.generate(
                    f"The user shows an image: {description}\nAssistant:",
                    emotion=emotion.describe(),
                )
                history.extend([f"Image: {description}", f"Assistant: {response}"])
                core.remember([f"Image: {description}", f"Assistant: {response}"])
                output.speak(response)
            except Exception as exc:
                output.speak(f"Could not analyze image: {exc}")
            continue
        if text.lower() == "summary":
            if summarizer is None:
                summarizer = Summarizer()
            summary = summarizer.summarize(history)
            output.speak(f"Summary: {summary}")
            continue
        emotion.update(text)
        intent = planner.analyze(text)
        response = core.generate(
            f"User: {intent}\nAssistant:", emotion=emotion.describe()
        )
        history.extend([f"User: {text}", f"Assistant: {response}"])
        core.remember([f"User: {text}", f"Assistant: {response}"])
        output.speak(response)
        print(f"(Emotion: {emotion.describe()})")


if __name__ == "__main__":
    main()
