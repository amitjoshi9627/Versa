from typing import Generator
from unittest.mock import patch

import pytest

from chatbot.engine import BaseChatBotEngine, ChatBotEngine
from chatbot.model import ModelLoader
from tests.constants import CHILD, COMEDIAN, DOCBOT, LLM_MODEL


@pytest.fixture
def chatbot_engine() -> Generator[ChatBotEngine, None, None]:
    """Provides a mocked ChatBotEngine instance for testing.

    Yields:
        ChatBotEngine: A mocked ChatBotEngine instance.
    """

    mock_model = "mocked_model"
    mock_tokenizer = "mocked_tokenizer"

    with patch.object(ModelLoader, "load", return_value=(mock_model, mock_tokenizer)):
        chatbot_engine = ChatBotEngine(LLM_MODEL)
        yield chatbot_engine


@pytest.fixture
def base_chatbot_engine() -> Generator[BaseChatBotEngine, None, None]:
    """Provides a mocked BaseChatBotEngine instance for testing.

    Yields:
        BaseChatBotEngine: A mocked BaseChatBotEngine instance.
    """

    mock_model = "mocked_model"
    mock_tokenizer = "mocked_tokenizer"

    with patch.object(ModelLoader, "load", return_value=(mock_model, mock_tokenizer)):
        chatbot_engine = BaseChatBotEngine(LLM_MODEL)
        yield chatbot_engine


class TestEngine:
    """Test class for testing the ChatBotEngine and BaseChatBotEngine classes."""

    @pytest.mark.parametrize("chatbot_type", [CHILD, COMEDIAN])
    def test__verify_chatbot_type_pass(self, chatbot_engine: ChatBotEngine, chatbot_type: str) -> None:
        """Tests the `verify_chatbot_type` method with valid chatbot types.

        Args:
            chatbot_engine (ChatBotEngine): The ChatBotEngine instance.
            chatbot_type (str): The chatbot type to test.
        """

        assert chatbot_engine.verify_chatbot_type(chatbot_type) == chatbot_type

    @pytest.mark.parametrize("chatbot_type", ["random_type", DOCBOT])
    def test__verify_chatbot_type_fail(self, chatbot_engine: ChatBotEngine, chatbot_type: str) -> None:
        """Tests the `verify_chatbot_type` method with invalid chatbot types.

        Args:
            chatbot_engine (ChatBotEngine): The ChatBotEngine instance.
            chatbot_type (str): The invalid chatbot type to test.
        """

        with pytest.raises(ValueError):
            chatbot_engine.verify_chatbot_type(chatbot_type)

    @pytest.mark.parametrize(
        "response",
        [
            [{"generated_text": " Hi! I am good! What about you?"}],
            [{"generated_text": "Did you knew Blue whale is a mammal?"}],
        ],
    )
    def test__format_response(self, base_chatbot_engine: BaseChatBotEngine, response: list[dict[str, str]]) -> None:
        """Tests the `_format_response` method.

        Args:
            base_chatbot_engine (BaseChatBotEngine): The BaseChatBotEngine instance.
            response (list[dict[str, str]]): The LLM response to test.
        """

        formatted_response = base_chatbot_engine._format_response(response)

        assert formatted_response == response[0]["generated_text"]
