from chatbot.constants import PDF_FILE_PATH, DOCBOT, CHAT_SEPARATOR
from chatbot.engine import ChatbotEngine

if __name__ == "__main__":
    # `chatbot_type` is `Docbot` to run chatbot with doc
    chat_bot = ChatbotEngine(chatbot_type=DOCBOT, file_path=PDF_FILE_PATH)
    question = "What is the typical size of a blue whale?"
    response_message = chat_bot.get_response(query=question)
    print(
        f"{CHAT_SEPARATOR}Question: {response_message.query}{CHAT_SEPARATOR}Response: {response_message.response}"
    )
