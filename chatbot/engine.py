from typing import Optional

from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.vectorstores import FAISS
from transformers import Pipeline, pipeline

from chatbot.constants import (
    CHAT_SEPARATOR,
    CHATBOT_TYPE_LIST,
    DEFAULT,
    DEFAULT_LLM_TEMP,
    DOCBOT,
    MACOS,
    MAX_NEW_TOKENS,
)
from chatbot.login import login_user
from chatbot.model import (
    ModelLoader,
)
from chatbot.preprocessing import load_data, remove_duplicate, split_item
from chatbot.prompt import (
    PERSONALITY_PROMPTS,
    PromptGenerator,
)
from chatbot.retriever import (
    retrieve_docs,
)
from chatbot.utils import ResponseMessage, get_os
from chatbot.vector_database import (
    get_vector_database,
)


class BaseChatBotEngine:
    def __init__(
        self,
        model_name_or_path: str,
        quantize: bool = False,
        llm_temp: float = DEFAULT_LLM_TEMP,
        access_token: Optional[str] = None,
    ):
        self.os = get_os()
        if access_token:
            login_user(access_token=access_token)
        self.llm_model, self.tokenizer = ModelLoader.load(model_name_or_path, quantize)
        self.prompt_generator = PromptGenerator()
        self.prompts = PERSONALITY_PROMPTS
        self.llm_temperature = llm_temp

    def get_pipeline(self) -> Pipeline | MLXPipeline:
        if self.os == MACOS:
            return MLXPipeline(
                model=self.llm_model,
                tokenizer=self.tokenizer,
                pipeline_kwargs={
                    "temp": self.llm_temperature,
                    "max_tokens": MAX_NEW_TOKENS,
                    "repetition_penalty": 1.1,
                },
            )
        else:
            return pipeline(
                model=self.llm_model,
                tokenizer=self.tokenizer,
                task="text-generation",
                do_sample=True,
                temperature=self.llm_temperature,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

    def get_prompt_template(self, chatbot_type: str) -> str:
        prompt = self.prompt_generator.generate(
            self.prompts[chatbot_type],
            with_summary=False,
            with_history=False,
        )
        prompt_template = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        return prompt_template

    @staticmethod
    def _format_response(response: list[dict[str, str]]) -> str:
        return response[0]["generated_text"]

    def get_response(self, query: str) -> ResponseMessage:
        llm_pipeline = self.get_pipeline()
        response = (
            llm_pipeline(query) if self.os == MACOS else self._format_response(llm_pipeline(query))
        ).strip()
        return ResponseMessage(query=query, response=response)


class DocBotEngine(BaseChatBotEngine):
    def __init__(
        self,
        file_path: str,
        model_name_or_path: str,
        quantize: bool = False,
        llm_temp: float = DEFAULT_LLM_TEMP,
        access_token: Optional[str] = None,
    ):
        """ChatBotEngine.

        Args:
            file_path: File path for processing.
            model_name_or_path: model name or path
            quantize: whether to quantize model or not.
            llm_temp: Parameter influencing the balance between predictability
                      and creativity in generated text
                     (less than 1 for more deterministic or greater than 1 for more creative)
            access_token: (Optional). Access token for gated models in huggingface
        """
        super().__init__(model_name_or_path, quantize, llm_temp, access_token)
        self.chatbot_type = DOCBOT
        self.vec_database: FAISS = self.process_doc(file_path)

    @staticmethod
    def process_doc(file_path: str) -> FAISS:
        content = load_data(file_path)
        splitted_items = split_item(content)
        unique_docs = remove_duplicate(splitted_items)
        return get_vector_database(unique_docs)

    def retriever(self, query: str) -> list[str]:
        return retrieve_docs(
            query=query,
            knowledge_index=self.vec_database,
        )

    def process_prompt(self, query: str) -> str:
        prompt_template = self.get_prompt_template(self.chatbot_type)
        relevant_docs = self.retriever(query)
        context = f"{CHAT_SEPARATOR}Extracted documents:{CHAT_SEPARATOR}"
        context += "".join(
            [
                f"Document {str(ind)}:::{CHAT_SEPARATOR}" + doc
                for ind, doc in enumerate(relevant_docs)
            ]
        )
        return prompt_template.format(query=query, context=context)

    def get_response(self, query: str) -> ResponseMessage:
        prompt = self.process_prompt(query=query)
        response_message = super().get_response(prompt)
        return ResponseMessage(query=query, response=response_message.response)


class ChatBotEngine(BaseChatBotEngine):
    def __init__(
        self,
        model_name_or_path: str,
        quantize: bool = False,
        chatbot_type: str = DEFAULT,
        llm_temp: float = DEFAULT_LLM_TEMP,
        access_token: Optional[str] = None,
    ):
        """ChatBotEngine.

        Args:
            model_name_or_path : model name or path.
            quantize: whether to quantize model or not.
            chatbot_type: Type of chatbot to load - Available Options
                            {'Therapist', 'Comedian', 'Default', 'Child', 'Expert'}
            llm_temp: Parameter influencing the balance between predictability
                      and creativity in generated text
                     (less than 1 for more deterministic or greater than 1 for more creative)
            access_token: (Optional). Access token for gated models in huggingface
        """
        super().__init__(model_name_or_path, quantize, llm_temp, access_token)
        self.chatbot_type = self.verify_chatbot_type(chatbot_type)

    @staticmethod
    def verify_chatbot_type(chatbot_type: str) -> str:
        if chatbot_type not in CHATBOT_TYPE_LIST:
            raise ValueError(
                f"Chatbot type `{chatbot_type}` is not supported. Choose from - {CHATBOT_TYPE_LIST}"
            )

        return chatbot_type

    def process_prompt(self, query: str) -> str:
        prompt_template = self.get_prompt_template(self.chatbot_type)
        return prompt_template.format(query=query)

    def get_response(self, query: str) -> ResponseMessage:
        prompt = self.process_prompt(query=query)
        response_message = super().get_response(prompt)
        return ResponseMessage(query=query, response=response_message.response)
