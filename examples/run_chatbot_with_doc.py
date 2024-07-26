from chatbot.constants import PDF_FILE_PATH
from chatbot.engine import ChatbotEngine

if __name__ == "__main__":
    chat_bot = ChatbotEngine(file_path=PDF_FILE_PATH)
    question = "What are some common threats to Environment?"
    response = chat_bot.get_response(query=question)
    print(f"\nQuestion: {question}\nResponse: {response}")
