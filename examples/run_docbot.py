from chatbot.constants import PDF_FILE_PATH, CHAT_SEPARATOR, EXAMPLE_LLM_MODEL, DETERMINISTIC_LLM_TEMP
from chatbot.engine import DocBotEngine

if __name__ == "__main__":

    # Use access token for gated huggingface models
    doc_bot = DocBotEngine(
        file_path=PDF_FILE_PATH,
        model_name_or_path=EXAMPLE_LLM_MODEL,
        quantize=False,
        llm_temp=DETERMINISTIC_LLM_TEMP,
    )
    question = "What is the typical size of a blue whale?"
    response_message = doc_bot.get_response(query=question)
    print(
        f"{CHAT_SEPARATOR}Question: {response_message.query}{CHAT_SEPARATOR}Response: {response_message.response}"
    )
