from chatbot.memory import ConversationBufferMemory
from chatbot.streamlit.utils import ChatMessage
from tests.constants import CHAT_SEPARATOR


class TestConversationMemory:
    def test_conversation_buffer_memory(self, conversation_history: list[ChatMessage]) -> None:
        memory = ConversationBufferMemory(buffer_len=4)
        result = memory.generate_history(conversation_history)
        assert isinstance(result, str)
        assert len(result.split(CHAT_SEPARATOR)) == len(conversation_history)
