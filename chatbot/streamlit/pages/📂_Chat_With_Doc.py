from typing import Iterator, TypeVar

import streamlit as st
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from chatbot.constants import ASSISTANT, CHAT_HISTORY, DATABASE, PDF_FILE_PATH, USER
from chatbot.data_preprocessing import load_data, remove_duplicate, split_item
from chatbot.memory import ConversationBufferMemory
from chatbot.prompt import CHATBOT_DOC_PROMPT_W_HISTORY
from chatbot.retriever import retrieve_docs
from chatbot.streamlit.utils import (
    ChatMessage,
    chat_history_init,
    clean_eos_token,
    load_llm_model,
    view_chat_history,
)
from chatbot.vector_database import get_vector_database

st.title("💭 Your Own Chatbot with Doc 🤖")

_CHAT_WITH_DOC = "chat_with_doc"

st.session_state.page = _CHAT_WITH_DOC
st.session_state[CHAT_HISTORY] = st.session_state.get(CHAT_HISTORY, [])

Output = TypeVar("Output")


@st.cache_resource(show_spinner=False)
def data_process(
    file_path: str = PDF_FILE_PATH,
) -> FAISS:
    content = load_data(file_path)
    splitted_items = split_item(content)
    unique_docs = remove_duplicate(splitted_items)

    vec_database = get_vector_database(unique_docs)

    return vec_database


class CustomDocChatbot:
    def __init__(self) -> None:
        with st.spinner("Loading Model..."):
            self.llm, self.tokenizer = load_llm_model()
        chat_history_init(_CHAT_WITH_DOC)
        self.avatar = {USER: "🐼", ASSISTANT: "🤖"}

    def get_response(self, query: str) -> Iterator[Output]:
        relevant_docs = retrieve_docs(
            query=query,
            knowledge_index=st.session_state[DATABASE],
        )
        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {str(ind)}:::\n" + doc for ind, doc in enumerate(relevant_docs)]
        )

        memory = ConversationBufferMemory(buffer_len=4)
        chat_history = memory.generate_history(st.session_state[CHAT_HISTORY])

        pipeline = MLXPipeline(
            model=self.llm,
            tokenizer=self.tokenizer,
            pipeline_kwargs={
                "temp": 0.7,
                "max_tokens": 520,
            },
        )

        prompt_template = self.tokenizer.apply_chat_template(
            CHATBOT_DOC_PROMPT_W_HISTORY,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)

        chain = prompt | pipeline | StrOutputParser()

        return chain.stream(
            {
                "question": query,
                "context": context,
                "history": chat_history,
            }
        )

    def get_user_input(self) -> None:
        if user_input := st.chat_input("Ask Me Anything!"):
            st.session_state[CHAT_HISTORY].append(ChatMessage(role=USER, message=user_input))
            with st.chat_message(USER, avatar=self.avatar[USER]):
                st.markdown(user_input)

            with st.chat_message(ASSISTANT, avatar=self.avatar[ASSISTANT]):
                response = clean_eos_token(
                    st.write_stream(self.get_response(user_input)), eos_token="</s>"
                )
            st.session_state[CHAT_HISTORY].append(ChatMessage(role=ASSISTANT, message=response))

    def run(self) -> None:
        with st.sidebar:
            st.title("Upload a file")
            pdf_doc = st.file_uploader("Upload a file")
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    st.session_state[DATABASE] = data_process(pdf_doc)
                    st.success("File processed.")

        view_chat_history()
        self.get_user_input()


if __name__ == "__main__":
    chatbot = CustomDocChatbot()
    chatbot.run()
