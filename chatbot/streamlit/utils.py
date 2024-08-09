from dataclasses import dataclass

import streamlit as st

from chatbot.constants import CHAT_HISTORY, CHAT_SEPARATOR, CHATBOT_TYPE, LLM_MODEL
from chatbot.model import ModelLoader


@dataclass
class ChatMessage:
    role: str
    message: str


def chat_history_init(current_chatbot_type: str) -> None:
    if st.session_state[CHATBOT_TYPE] != current_chatbot_type:
        del st.session_state[CHAT_HISTORY]
        st.session_state[CHAT_HISTORY] = list()
        st.session_state[CHATBOT_TYPE] = current_chatbot_type


def view_chat_history(avatar: dict[str, str]) -> None:
    for chat_message in st.session_state[CHAT_HISTORY]:
        with st.chat_message(chat_message.role, avatar=avatar[chat_message.role]):
            st.markdown(chat_message.message)


def chat_history_to_str(conversation: list[ChatMessage]) -> str:
    conversation_history = ""
    for chat_message in conversation:
        conversation_history += f"{chat_message.role}: {chat_message.message}{CHAT_SEPARATOR}"
    return conversation_history


@st.cache_resource(show_spinner=False)
def load_llm_model() -> tuple:
    model, tokenizer = ModelLoader.load(LLM_MODEL)
    return model, tokenizer
