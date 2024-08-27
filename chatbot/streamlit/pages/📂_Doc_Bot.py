import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT_DIR))


import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from chatbot.constants import (
    ASSISTANT,
    CHAT_HISTORY,
    CHAT_SEPARATOR,
    CHATBOT_TYPE,
    DATABASE,
    DETERMINISTIC_LLM_TEMP,
    DOCBOT,
    PDF_FILE_PATH,
)
from chatbot.preprocessing import load_data, remove_duplicate, split_item
from chatbot.retriever import retrieve_docs
from chatbot.streamlit.constants import START_VERSA
from chatbot.streamlit.engine import StreamlitEngine
from chatbot.streamlit.utils import (
    view_chat_history,
)
from chatbot.vector_database import get_vector_database

st.set_page_config(
    page_title="Versa: Your Personal AI Companion",
    page_icon="🤖",
    layout="wide",
)


st.markdown(
    """
    <h1 style='text-align: center; letter-spacing: 0.015em;
     font-family: Montserrat, sans-serif; font-weight: 500; '>
    ✨ Versa: Your Personal AI Companion ✨
    </h1>""",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <h3 style="text-align: center; font-weight: 500;
     letter-spacing: 0.075em; font-family: Montserrat, sans-serif;">
        Doc Bot 📂<br>
        <span style="font-size: 0.75em; font-weight: 300;
         letter-spacing: 0.1em; font-family: Montserrat, sans-serif;">
        Query - find - done!
        </span>
    </h3>
    """,
    unsafe_allow_html=True,
)

st.session_state[CHAT_HISTORY] = st.session_state.get(CHAT_HISTORY, [])
st.session_state[CHATBOT_TYPE] = st.session_state.get(CHATBOT_TYPE, None)


@st.cache_resource(show_spinner=False)
def data_process(
    file_path: str = PDF_FILE_PATH,
) -> FAISS:
    content = load_data(file_path)
    splitted_items = split_item(content)
    unique_docs = remove_duplicate(splitted_items)

    vec_database = get_vector_database(unique_docs)

    return vec_database


class StreamlitDocBotEngine(StreamlitEngine):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _verify_data_processed() -> bool:
        if DATABASE not in st.session_state:
            st.error("Did you forget to Submit & Process file in the side panel?")
            return False
        return True

    def get_response(self, chatbot_type: str, query: str) -> str:
        if not self._verify_data_processed():
            return ""
        pipeline = self.get_pipeline(llm_temp=DETERMINISTIC_LLM_TEMP)

        relevant_docs = retrieve_docs(
            query=query,
            knowledge_index=st.session_state[DATABASE],
        )

        context = f"{CHAT_SEPARATOR}Extracted documents:{CHAT_SEPARATOR}"
        context += "".join(
            [
                f"Document {str(ind)}:::{CHAT_SEPARATOR}" + doc
                for ind, doc in enumerate(relevant_docs)
            ]
        )

        memory = self.get_memory(memory_type="buffer")
        chat_history = memory.generate_history(st.session_state[CHAT_HISTORY])

        prompt = ChatPromptTemplate.from_template(
            self.get_prompt_template(
                chatbot_type=chatbot_type, with_summary=False, with_history=True
            )
        )

        chain = prompt | pipeline | StrOutputParser()

        return chain.invoke(
            {
                "query": query,
                "context": context,
                "history": chat_history,
            }
        )

    def run(self) -> None:
        if st.session_state.get(START_VERSA):
            chatbot_type = DOCBOT
            with st.sidebar:
                st.title("Upload a file")
                pdf_doc = st.file_uploader("Upload a file")
                if st.button("Submit & Process"):
                    if pdf_doc:
                        with st.spinner("Processing..."):
                            st.session_state[DATABASE] = data_process(pdf_doc)
                            self._display_call_out(
                                "File processed.",
                                icon="✅",
                                call_out_type="success",
                                wait_time=2.5,
                            )
                    else:
                        self._display_call_out(
                            "Please upload a file before submitting!",
                            icon="⚠️",
                            call_out_type="warning",
                            wait_time=2.5,
                        )

            self.avatars[ASSISTANT] = self.avatars[chatbot_type]
            self.change_chatbot_type(chatbot_type)
            view_chat_history(self.avatars)
            self.get_user_input(chatbot_type=chatbot_type)


if __name__ == "__main__":
    chatbot = StreamlitDocBotEngine()
    chatbot.run()
