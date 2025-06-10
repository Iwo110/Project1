from chatbot import ChatBot

if __name__ == "__main__":
    bot = ChatBot()
    print("Interactive chat session. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = bot.chat(user_input)
        print(f"Bot: {response}")
