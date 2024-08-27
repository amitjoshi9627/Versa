import pytest

from chatbot.engine import ChatBotEngine, DocBotEngine
from chatbot.utils import ResponseMessage
from tests.constants import CHILD, COMEDIAN, LLM_MODEL, TEST_PDF_FILE_PATH


def get_chatbot_engine(chatbot_type: str) -> ChatBotEngine:
    return ChatBotEngine(LLM_MODEL, chatbot_type=chatbot_type)


def get_docbot_engine(file_path: str) -> DocBotEngine:
    return DocBotEngine(file_path, LLM_MODEL)


@pytest.mark.parametrize("chatbot_type", [CHILD, COMEDIAN])
def test_chatbot_engine_integration(chatbot_type: str) -> None:
    chatbot_engine = get_chatbot_engine(chatbot_type)
    response_message = chatbot_engine.get_response("What is the greatest life lesson?")
    assert isinstance(response_message, ResponseMessage)
    assert isinstance(response_message.response, str)
    assert len(response_message.response) > 0


@pytest.mark.parametrize("file_path", [TEST_PDF_FILE_PATH])
def test_docbot_engine_integration(file_path: str) -> None:
    docbot_engine = get_docbot_engine(file_path)
    response_message = docbot_engine.get_response("What is a blue whale?")
    assert isinstance(response_message, ResponseMessage)
    assert isinstance(response_message.response, str)
    assert len(response_message.response) > 0
