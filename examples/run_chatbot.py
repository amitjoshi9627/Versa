from chatbot.constants import DEFAULT, CHAT_SEPARATOR, LLM_MODEL, CREATIVE_LLM_TEMP
from chatbot.engine import ChatBotEngine

if __name__ == "__main__":
    # Available `chatbot_type` - {'Therapist', 'Comedian', 'Default', 'Child', 'Expert'}

    chat_bot = ChatBotEngine(
        chatbot_type=DEFAULT,
        model_name_or_path=LLM_MODEL,
        quantize=False,
        llm_temp=CREATIVE_LLM_TEMP,
    )
    question = "What is 2/2?"
    response_message = chat_bot.get_response(query=question)
    print(
        f"{CHAT_SEPARATOR}Question: {response_message.query}{CHAT_SEPARATOR}Response: {response_message.response}"
    )
