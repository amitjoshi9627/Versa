from langchain_core.prompts import PromptTemplate
from mlx_lm import generate
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from chatbot.constants import USER
from chatbot.prompt import SUMMARIZATION_PROMPT
from chatbot.streamlit.utils import ChatMessage, chat_history_to_str


class ConversationSummaryBufferMemory:
    def __init__(
        self,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        buffer_len: int = 6,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.buffer_len = buffer_len
        self.summary_prompt = PromptTemplate.from_template(SUMMARIZATION_PROMPT)

    def summarise_conversation(self, conversation: list[ChatMessage]) -> str:
        if len(conversation) == 0:
            return ""

        return generate(
            self.llm,
            self.tokenizer,
            prompt=self.summary_prompt.format(text=chat_history_to_str(conversation)),
            verbose=True,
        )

    def generate_history(self, conversation: list[ChatMessage]) -> tuple[str, str]:
        if len(conversation) > 0 and conversation[-1].role == USER:
            conversation = conversation[:-1]  # Removing the latest query by user

        conversation_buffer_chat: list = list()
        buffer_len = min(self.buffer_len, len(conversation))
        conversation_buffer_text = chat_history_to_str(conversation_buffer_chat[-buffer_len:])
        conversation_summary_text = self.summarise_conversation(
            conversation[: len(conversation) - buffer_len]
        )
        return conversation_summary_text, conversation_buffer_text


class ConversationBufferMemory:
    def __init__(self, buffer_len: int = 6):
        self.buffer_len = buffer_len

    def generate_history(self, conversation: list[ChatMessage]) -> str:
        if len(conversation) > 0 and conversation[-1].role == USER:
            conversation = conversation[:-1]  # Removing the latest query by user

        return chat_history_to_str(conversation[-self.buffer_len :])
