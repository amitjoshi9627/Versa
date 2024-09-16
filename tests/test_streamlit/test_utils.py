from chatbot.streamlit.utils import ChatMessage, chat_history_to_str
from tests.constants import CHAT_SEPARATOR


def test_chat_history_to_str(conversation_history: list[ChatMessage]) -> None:
    """Tests the `chat_history_to_str` function to ensure it correctly converts the conversation
    history to a string.

    Args:
        conversation_history (list[ChatMessage]): The conversation history to test.
    """

    result = chat_history_to_str(conversation_history)
    assert isinstance(result, str)
    assert len(result.split(CHAT_SEPARATOR)) == len(conversation_history) + 1
