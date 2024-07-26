from chatbot.engine import ChatbotEngine

if __name__ == "__main__":
    chat_bot = ChatbotEngine()
    question = "What is 2/2?"
    response = chat_bot.get_response(query=question)
    print(f"\nQuestion: {question}\nResponse: {response}")
