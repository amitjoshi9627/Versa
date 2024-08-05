import pytest

from chatbot.streamlit.utils import ChatMessage
from tests.constants import ASSISTANT, USER


@pytest.fixture
def conversation_history() -> list[ChatMessage]:
    return [
        ChatMessage(role=USER, message="User's initial message"),
        ChatMessage(role=ASSISTANT, message="Assistant's Initial response"),
        ChatMessage(role=USER, message="User's Question?"),
        ChatMessage(role=ASSISTANT, message="Assistant's response."),
        ChatMessage(role=USER, message="User's appreciation."),
    ]
