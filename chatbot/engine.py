from typing import Optional

from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.vectorstores import FAISS
from transformers import (
    Pipeline,
)

from chatbot.constants import (
    CHAT_SEPARATOR,
    CHATBOT_TYPE_LIST,
    CREATIVE_LLM_TEMP,
    DEFAULT,
    DETERMINISTIC_LLM_TEMP,
    DOCBOT,
    MACOS,
    MAX_NEW_TOKENS,
)
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


class ChatbotEngine:
    def __init__(self, chatbot_type: str = DEFAULT, file_path: Optional[str] = None):
        """ChatbotEngine.

        Args:
            chatbot_type: Type of chatbot to load - Available Options
                            {'Therapist', 'Comedian', 'Default', 'Child', 'Expert'}
            file_path: Optional , file_path for chatbot with document
        """
        self.os = get_os()
        if file_path:
            self.vec_database: FAISS = self.process_doc(file_path)
        self.with_doc = bool(file_path)
        self.chatbot_type = self.verify_chatbot_type(chatbot_type)
        self.llm_model, self.tokenizer = ModelLoader.load()
        self.prompt_generator = PromptGenerator()
        self.prompts = PERSONALITY_PROMPTS

    def verify_chatbot_type(self, chatbot_type: str) -> str:
        if chatbot_type not in CHATBOT_TYPE_LIST:
            raise ValueError(
                f"Chatbot type `{chatbot_type}` is not supported. Choose from - {CHATBOT_TYPE_LIST}"
            )
        if chatbot_type == DOCBOT and not self.with_doc:
            raise ValueError(f"Please provide `file_path` with `chatbot_type` = {DOCBOT}")
        return chatbot_type

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

    def get_pipeline(self) -> Pipeline | MLXPipeline:
        llm_temperature = DETERMINISTIC_LLM_TEMP if self.with_doc else CREATIVE_LLM_TEMP
        if self.os == MACOS:
            return MLXPipeline(
                model=self.llm_model,
                tokenizer=self.tokenizer,
                pipeline_kwargs={
                    "temp": llm_temperature,
                    "max_tokens": MAX_NEW_TOKENS,
                    "repetition_penalty": 1.1,
                },
            )
        else:
            return Pipeline(
                model=self.llm_model,
                tokenizer=self.tokenizer,
                task="text-generation",
                do_sample=True,
                temperature=llm_temperature,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

    def process_prompt(self, query: str) -> str:
        prompt = self.prompt_generator.generate(
            self.prompts[self.chatbot_type],
            with_summary=False,
            with_history=False,
        )
        prompt_template = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        if self.with_doc:
            relevant_docs = self.retriever(query)
            context = f"{CHAT_SEPARATOR}Extracted documents:{CHAT_SEPARATOR}"
            context += "".join(
                [
                    f"Document {str(ind)}:::{CHAT_SEPARATOR}" + doc
                    for ind, doc in enumerate(relevant_docs)
                ]
            )
            return prompt_template.format(query=query, context=context)
        else:
            return prompt_template.format(query=query)

    def get_response(self, query: str) -> ResponseMessage:
        prompt = self.process_prompt(query=query)
        pipeline = self.get_pipeline()
        return ResponseMessage(query=query, response=pipeline(prompt))
