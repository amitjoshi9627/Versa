from typing import Generator
from unittest.mock import patch

import pytest

from chatbot.engine import ChatbotEngine
from chatbot.model import ModelLoader
from tests.constants import CHILD, COMEDIAN


@pytest.fixture
def chatbot_engine() -> Generator[ChatbotEngine, None, None]:
    mock_model = "mocked_model"
    mock_tokenizer = "mocked_tokenizer"

    with patch.object(ModelLoader, "load", return_value=(mock_model, mock_tokenizer)):
        chatbot_engine = ChatbotEngine()
        yield chatbot_engine


class TestEngine:
    @pytest.mark.parametrize("chatbot_type", [CHILD, COMEDIAN])
    def test__verify_chatbot_type_pass(
        self, chatbot_engine: ChatbotEngine, chatbot_type: str
    ) -> None:
        assert chatbot_engine.verify_chatbot_type(chatbot_type) == chatbot_type

    @pytest.mark.parametrize("chatbot_type", ["random_type"])
    def test__verify_chatbot_type_fail(
        self, chatbot_engine: ChatbotEngine, chatbot_type: str
    ) -> None:
        with pytest.raises(ValueError):
            chatbot_engine.verify_chatbot_type(chatbot_type)
