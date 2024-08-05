import pytest

from chatbot.streamlit.utils import ChatMessage, chat_history_to_str, clean_eos_token
from tests.constants import CHAT_SEPARATOR, EOS_TOKEN


@pytest.mark.parametrize(
    "message", [f"Sample message with eos token {EOS_TOKEN}", "Sample message with no eos token"]
)
def test_clean_eos_token(message: str) -> None:
    result = clean_eos_token(message, eos_token=EOS_TOKEN)
    assert result[-len(EOS_TOKEN) :] != EOS_TOKEN


def test_chat_history_to_str(conversation_history: list[ChatMessage]) -> None:
    result = chat_history_to_str(conversation_history)
    assert isinstance(result, str)
    assert len(result.split(CHAT_SEPARATOR)) == len(conversation_history) + 1
