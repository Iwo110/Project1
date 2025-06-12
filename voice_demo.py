"""Voice-enabled demo using speech recognition and speech synthesis."""

from perception import SpeechRecognizer
from cognitive import CognitiveCore
from emotion import EmotionState
from planning import Planner
from effector import SpeechSynthesizer
from moderation import SafetyChecker


def main() -> None:
    recognizer = SpeechRecognizer()
    core = CognitiveCore(
        vector_memory_path="voice_memory", memory_file="voice_history.txt"
    )
    emotion = EmotionState(mood_file="voice_mood.txt")
    planner = Planner()
    speaker = SpeechSynthesizer()
    checker = SafetyChecker()
    if core.memory:
        history = core.memory.load()
        if core.vector_memory and not core.vector_memory.index_path.exists():
            for line in history:
                core.vector_memory.add(line)

    print("Voice demo. Say something or press Ctrl+C to exit.")
    try:
        while True:
            text = recognizer.listen(timeout=5)
            if not text:
                continue
            print(f"You said: {text}")
            emotion.update(text)
            intent = planner.analyze(text)
            response = core.generate(
                f"User: {intent}\nAssistant:", emotion=emotion.describe()
            )
            if checker.is_toxic(response):
                response = "Przepraszam, ale wolę nie mówić na ten temat."
            core.remember([f"User: {text}", f"Assistant: {response}"])
            speaker.say(response)
    except KeyboardInterrupt:
        print("Exiting.")


if __name__ == "__main__":
    main()
