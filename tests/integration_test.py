import pytest

from chatbot.engine import ChatBotEngine, DocBotEngine
from chatbot.utils import ResponseMessage
from tests.constants import CHILD, COMEDIAN, LLM_MODEL, TEST_PDF_FILE_PATH


def get_chatbot_engine(chatbot_type: str) -> ChatBotEngine:
    """Gets a ChatBotEngine instance for the specified chatbot type.

    Args:
        chatbot_type (str): The type of chatbot.

    Returns:
        ChatBotEngine: The ChatBotEngine instance.
    """

    return ChatBotEngine(LLM_MODEL, chatbot_type=chatbot_type)


def get_docbot_engine(file_path: str) -> DocBotEngine:
    """Gets a DocBotEngine instance for the specified file path.

    Args:
        file_path (str): The path to the document file.

    Returns:
        DocBotEngine: The DocBotEngine instance.
    """

    return DocBotEngine(file_path, LLM_MODEL)


@pytest.mark.parametrize("chatbot_type", [CHILD, COMEDIAN])
def test_chatbot_engine_integration(chatbot_type: str) -> None:
    """Tests the integration of the ChatBotEngine with different chatbot types.

    Args:
        chatbot_type (str): The type of chatbot to test.
    """

    chatbot_engine = get_chatbot_engine(chatbot_type)
    response_message = chatbot_engine.get_response("What is the greatest life lesson?")
    assert isinstance(response_message, ResponseMessage)
    assert isinstance(response_message.response, str)
    assert len(response_message.response) > 0


@pytest.mark.parametrize("file_path", [TEST_PDF_FILE_PATH])
def test_docbot_engine_integration(file_path: str) -> None:
    """Tests the integration of the DocBotEngine with a document file.

    Args:
        file_path (str): The path to the document file.
    """

    docbot_engine = get_docbot_engine(file_path)
    response_message = docbot_engine.get_response("What is a blue whale?")
    assert isinstance(response_message, ResponseMessage)
    assert isinstance(response_message.response, str)
    assert len(response_message.response) > 0
