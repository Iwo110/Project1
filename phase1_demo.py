"""Demonstration of the Phase 1 architecture skeleton."""

from perception import TextInput
from cognitive import CognitiveCore
from emotion import EmotionState
from planning import Planner
from effector import TextOutput


def main() -> None:
    input_module = TextInput()
    core = CognitiveCore()
    emotion = EmotionState()
    planner = Planner()
    output = TextOutput()

    print("Phase 1 demo. Type 'exit' to quit.")
    while True:
        text = input_module.receive()
        if text.lower() == "exit":
            break
        emotion.update(text)
        intent = planner.analyze(text)
        response = core.generate(f"User: {intent}\nAssistant:")
        core.remember([f"User: {text}", f"Assistant: {response}"])
        output.speak(response)
        print(f"(Emotion: {emotion.describe()})")


if __name__ == "__main__":
    main()
