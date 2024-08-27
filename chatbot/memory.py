from abc import abstractmethod

from langchain_core.prompts import PromptTemplate
from mlx_lm import generate
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from chatbot.constants import DEFAULT_BUFFER_LEN, USER
from chatbot.prompt import SUMMARIZATION_PROMPT, PromptGenerator
from chatbot.streamlit.utils import ChatMessage, chat_history_to_str


class ConversationMemory:
    def __init__(self, buffer_len: int) -> None:
        self.buffer_len = buffer_len

    @abstractmethod
    def generate_history(self, conversation: list[ChatMessage]) -> tuple[str, str] | str:
        raise NotImplementedError("Please Implement this method")


class ConversationSummaryBufferMemory(ConversationMemory):
    def __init__(
        self,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        buffer_len: int = DEFAULT_BUFFER_LEN,
    ) -> None:
        super().__init__(buffer_len)
        self.llm = llm
        self.tokenizer = tokenizer
        self.summary_prompt = PromptTemplate.from_template(SUMMARIZATION_PROMPT)
        self.prompt_generator = PromptGenerator()

    def summarise_conversation(self, conversation: list[ChatMessage]) -> str:
        if len(conversation) == 0:
            return ""

        prompt_template = self.tokenizer.apply_chat_template(
            self.prompt_generator.generate(
                SUMMARIZATION_PROMPT,
                with_summary=False,
                with_history=True,
                with_query=False,
            ),
            tokenize=False,
            add_generation_prompt=True,
        )
        summary_prompt = PromptTemplate.from_template(prompt_template)
        return generate(
            self.llm,
            self.tokenizer,
            prompt=summary_prompt.format(history=chat_history_to_str(conversation)),
            verbose=False,
        )

    def generate_history(self, conversation: list[ChatMessage]) -> tuple[str, str]:
        if len(conversation) > 0 and conversation[-1].role == USER:
            conversation = conversation[:-1]  # Removing the latest query by user

        buffer_len = min(self.buffer_len, len(conversation))
        conversation_buffer_text = chat_history_to_str(conversation[-buffer_len:])
        conversation_summary_text = self.summarise_conversation(
            conversation[: len(conversation) - buffer_len]
        )
        return conversation_summary_text, conversation_buffer_text


class ConversationBufferMemory(ConversationMemory):
    def __init__(self, buffer_len: int = DEFAULT_BUFFER_LEN) -> None:
        super().__init__(buffer_len)

    def generate_history(self, conversation: list[ChatMessage]) -> str:
        if len(conversation) > 0 and conversation[-1].role == USER:
            conversation = conversation[:-1]  # Removing the latest query by user

        return chat_history_to_str(conversation[-self.buffer_len :])
