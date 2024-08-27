from chatbot.constants import DEFAULT, CHAT_SEPARATOR, EXAMPLE_LLM_MODEL, CREATIVE_LLM_TEMP
from chatbot.engine import ChatBotEngine

if __name__ == "__main__":
    # Available `chatbot_type` - {'Therapist', 'Comedian', 'Default', 'Child', 'Expert'}

    # Use access token for gated huggingface models
    chat_bot = ChatBotEngine(
        chatbot_type=DEFAULT,
        model_name_or_path=EXAMPLE_LLM_MODEL,
        quantize=False,
        llm_temp=CREATIVE_LLM_TEMP,
    )
    question = "What is the greatest life lesson?"
    response_message = chat_bot.get_response(query=question)
    print(
        f"{CHAT_SEPARATOR}Question: {response_message.query}{CHAT_SEPARATOR}Response: {response_message.response}"
    )
