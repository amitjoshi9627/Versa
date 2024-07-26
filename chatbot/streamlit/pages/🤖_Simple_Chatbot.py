from typing import Iterator, TypeVar

import streamlit as st
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from chatbot.constants import ASSISTANT, CHAT_HISTORY, USER
from chatbot.memory import ConversationSummaryBufferMemory
from chatbot.prompt import SIMPLE_CHATBOT_PROMPT_W_HISTORY
from chatbot.streamlit.utils import (
    ChatMessage,
    chat_history_init,
    clean_eos_token,
    load_llm_model,
    view_chat_history,
)

st.title("💭 Your Own Happy Chatbot 👽")

_SIMPLE_CHATBOT = "simple_chatbot"
st.session_state.page = _SIMPLE_CHATBOT

st.session_state[CHAT_HISTORY] = st.session_state.get(CHAT_HISTORY, [])

Output = TypeVar("Output")


class SimpleChatbot:
    def __init__(self) -> None:
        with st.spinner("Loading Model..."):
            self.llm, self.tokenizer = load_llm_model()
        chat_history_init(_SIMPLE_CHATBOT)
        self.avatar = {USER: "🐼", ASSISTANT: "🤖"}

    def get_response(self, query: str) -> Iterator[Output]:
        pipeline = MLXPipeline(
            model=self.llm,
            tokenizer=self.tokenizer,
            pipeline_kwargs={
                "temp": 1.0,
                "max_tokens": 520,
            },
        )
        memory = ConversationSummaryBufferMemory(llm=self.llm, tokenizer=self.tokenizer)
        chat_summary, chat_history = memory.generate_history(st.session_state[CHAT_HISTORY])

        prompt_template = self.tokenizer.apply_chat_template(
            SIMPLE_CHATBOT_PROMPT_W_HISTORY,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | pipeline | StrOutputParser()
        return chain.stream(
            {
                "question": query,
                "summary": chat_summary,
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
        view_chat_history()
        self.get_user_input()


if __name__ == "__main__":
    chatbot = SimpleChatbot()
    chatbot.run()
