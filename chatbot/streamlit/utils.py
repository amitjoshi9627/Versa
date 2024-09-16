from dataclasses import dataclass

import streamlit as st

from chatbot.constants import CHAT_HISTORY, CHAT_SEPARATOR, CHATBOT_TYPE, LLM_MODEL
from chatbot.model import ModelLoader


@dataclass
class ChatMessage:
    """Represents a single message in a chat conversation.

    Attributes:
        role (str): The role of the sender (e.g., "user", "assistant").
        message (str): The text content of the message.
    """

    role: str
    message: str


def chat_history_init(current_chatbot_type: str) -> None:
    """Initializes the chat history and updates the chatbot type.

    Args:
        current_chatbot_type (str): The current chatbot type.
    """

    if st.session_state[CHATBOT_TYPE] != current_chatbot_type:
        del st.session_state[CHAT_HISTORY]
        st.session_state[CHAT_HISTORY] = list()
        st.session_state[CHATBOT_TYPE] = current_chatbot_type


def view_chat_history(avatar: dict[str, str]) -> None:
    """Displays the chat history in the Streamlit app.

    Args:
        avatar (dict[str, str]): A dictionary mapping chatbot roles to their corresponding avatars.
    """

    for chat_message in st.session_state[CHAT_HISTORY]:
        with st.chat_message(chat_message.role, avatar=avatar[chat_message.role]):
            st.markdown(chat_message.message)


def chat_history_to_str(conversation: list[ChatMessage]) -> str:
    """Converts a list of ChatMessage objects into a string representation of the conversation
    history.

    Args:
        conversation (list[ChatMessage]): The list of ChatMessage objects.

    Returns:
        str: The conversation history as a string.
    """

    conversation_history = ""
    for chat_message in conversation:
        conversation_history += f"{chat_message.role}: {chat_message.message}{CHAT_SEPARATOR}"
    return conversation_history


@st.cache_resource(show_spinner=False)
def load_llm_model() -> tuple:
    """Loads the LLM model and its corresponding tokenizer.

    Returns:
        tuple: A tuple containing the loaded LLM model and tokenizer.
    """

    model, tokenizer = ModelLoader.load(LLM_MODEL, quantize=False)
    return model, tokenizer
