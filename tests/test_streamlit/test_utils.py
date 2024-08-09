from chatbot.streamlit.utils import ChatMessage, chat_history_to_str
from tests.constants import CHAT_SEPARATOR


def test_chat_history_to_str(conversation_history: list[ChatMessage]) -> None:
    result = chat_history_to_str(conversation_history)
    assert isinstance(result, str)
    assert len(result.split(CHAT_SEPARATOR)) == len(conversation_history) + 1
