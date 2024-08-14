from chatbot.constants import PDF_FILE_PATH, CHAT_SEPARATOR
from chatbot.engine import DocBotEngine

if __name__ == "__main__":
    chat_bot = DocBotEngine(file_path=PDF_FILE_PATH)
    question = "What is the typical size of a blue whale?"
    response_message = chat_bot.get_response(query=question)
    print(
        f"{CHAT_SEPARATOR}Question: {response_message.query}{CHAT_SEPARATOR}Response: {response_message.response}"
    )
