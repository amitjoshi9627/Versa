from typing import Optional

import pytest

from chatbot.engine import ChatbotEngine
from chatbot.utils import ResponseMessage
from tests.constants import CHILD, DOCBOT, TEST_PDF_FILE_PATH


def get_chatbot_engine(chatbot_type: str, file_path: Optional[str] = None) -> ChatbotEngine:
    return ChatbotEngine(chatbot_type, file_path)


@pytest.mark.parametrize("chatbot_type, file_path", [(CHILD, None), (DOCBOT, TEST_PDF_FILE_PATH)])
def test_chatbot_engine_integration(chatbot_type: str, file_path: Optional[str] = None) -> None:
    chatbot_engine = get_chatbot_engine(chatbot_type, file_path)
    response_message = chatbot_engine.get_response("What is a blue whale?")
    assert isinstance(response_message, ResponseMessage)
    assert isinstance(response_message.response, str)
    assert len(response_message.response) > 0
