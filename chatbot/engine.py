from typing import Optional

from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.vectorstores import FAISS
from transformers import (
    Pipeline,
)

from chatbot.constants import (
    CREATIVE_LLM_TEMP,
    DETERMINISTIC_LLM_TEMP,
    MACOS,
    MAX_NEW_TOKENS,
)
from chatbot.data_preprocessing import load_data, remove_duplicate, split_item
from chatbot.model import (
    ModelLoader,
)
from chatbot.prompt import CHATBOT_DOC_PROMPT, SIMPLE_CHATBOT_PROMPT
from chatbot.retriever import (
    retrieve_docs,
)
from chatbot.utils import get_os
from chatbot.vector_database import (
    get_vector_database,
)


class ChatbotEngine:
    def __init__(self, file_path: Optional[str] = None):
        self.os = get_os()
        if file_path:
            self.vec_database: FAISS = self.process_doc(file_path)
        self.llm_model, self.tokenizer = ModelLoader.load()
        self.with_doc = bool(file_path)

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
        if self.with_doc:
            prompt_template = self.tokenizer.apply_chat_template(
                CHATBOT_DOC_PROMPT,
                tokenize=False,
                add_generation_prompt=True,
            )
            relevant_docs = self.retriever(query)
            context = "\nExtracted documents:\n"
            context += "".join(
                [f"Document {str(ind)}:::\n" + doc for ind, doc in enumerate(relevant_docs)]
            )
            return prompt_template.format(question=query, context=context)
        else:
            return self.tokenizer.apply_chat_template(
                SIMPLE_CHATBOT_PROMPT, tokenize=False, add_generation_prompt=True
            ).format(question=query)

    def get_response(self, query: str) -> str:
        prompt = self.process_prompt(query=query)
        pipeline = self.get_pipeline()

        return pipeline(prompt)
