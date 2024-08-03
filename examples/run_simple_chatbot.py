from chatbot.constants import DEFAULT
from chatbot.engine import ChatbotEngine

if __name__ == "__main__":
    # Available `chatbot_type` - {'Therapist', 'Comedian', 'Default', 'Child', 'Expert'}
    chat_bot = ChatbotEngine(chatbot_type=DEFAULT)
    question = "What is 2/2?"
    response_message = chat_bot.get_response(query=question)
    print(f"\nQuestion: {response_message.query}\nResponse: {response_message.response}")
