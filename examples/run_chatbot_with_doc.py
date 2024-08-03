from chatbot.constants import PDF_FILE_PATH, DOCBOT
from chatbot.engine import ChatbotEngine

if __name__ == "__main__":
    # `chatbot_type` is `Docbot` to run chatbot with doc
    chat_bot = ChatbotEngine(chatbot_type=DOCBOT, file_path=PDF_FILE_PATH)
    question = "What are some common threats to Environment?"
    response_message = chat_bot.get_response(query=question)
    print(f"\nQuestion: {response_message.query}\nResponse: {response_message.response}")
