from typing import Iterator, TypeVar

import streamlit as st
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from chatbot.constants import (
    ASSISTANT,
    BUFFER_LEN,
    CHAT_HISTORY,
    CHATBOT_TYPE,
    CHILD,
    COMEDIAN,
    CREATIVE_LLM_TEMP,
    DEFAULT,
    EOS_TOKEN,
    EXPERT,
    MAX_NEW_TOKENS,
    THERAPIST,
    USER,
)
from chatbot.memory import ConversationSummaryBufferMemory
from chatbot.prompt import (
    PERSONALITY_PROMPTS,
    SIMPLE_CHATBOT_PROMPT,
    PromptGenerator,
)
from chatbot.streamlit.utils import (
    ChatMessage,
    chat_history_init,
    clean_eos_token,
    load_llm_model,
    view_chat_history,
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
    <h5 style='text-align: center; letter-spacing: 0.02em;
     font-family: Montserrat, sans-serif; font-weight: 300; '>
    Relaxed conversations, Real results!<br>
    </h5>""",
    unsafe_allow_html=True,
)

st.session_state[CHAT_HISTORY] = st.session_state.get(CHAT_HISTORY, [])
st.session_state[CHATBOT_TYPE] = st.session_state.get(CHATBOT_TYPE, None)

Output = TypeVar("Output")


class SimpleChatbot:
    def __init__(self) -> None:
        with st.spinner("Loading Model..."):
            self.llm, self.tokenizer = load_llm_model()
        self.personality_prompts = PERSONALITY_PROMPTS
        self.personality_avatars = {
            THERAPIST: "🧑‍⚕️",
            EXPERT: "💡",
            CHILD: "🍭",
            COMEDIAN: "🎭",
            DEFAULT: "🤖",
        }
        self.avatar = {USER: "🐼"}
        self.prompt_template = SIMPLE_CHATBOT_PROMPT
        self.prompt_generator = PromptGenerator()

    def get_response(self, query: str) -> Iterator[Output]:
        pipeline = MLXPipeline(
            model=self.llm,
            tokenizer=self.tokenizer,
            pipeline_kwargs={
                "temp": CREATIVE_LLM_TEMP,
                "max_tokens": MAX_NEW_TOKENS,
            },
        )

        memory = ConversationSummaryBufferMemory(
            llm=self.llm, tokenizer=self.tokenizer, buffer_len=BUFFER_LEN
        )
        chat_summary, chat_history = memory.generate_history(st.session_state[CHAT_HISTORY])

        prompt_template = self.tokenizer.apply_chat_template(
            self.prompt_generator.generate(
                self.personality_prompts[st.session_state[CHATBOT_TYPE]],
                with_summary=True,
                with_history=True,
            ),
            tokenize=False,
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | pipeline | StrOutputParser()
        return chain.stream(
            {
                "query": query,
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
                    st.write_stream(self.get_response(user_input)), eos_token=EOS_TOKEN
                )
            st.session_state[CHAT_HISTORY].append(ChatMessage(role=ASSISTANT, message=response))

    @staticmethod
    def change_chatbot_type(chatbot_type: str) -> None:
        if st.session_state[CHATBOT_TYPE] != chatbot_type:
            chat_history_init(chatbot_type)

    def run(self) -> None:
        st.sidebar.title("Choose Your Personality")

        selected_personality = st.sidebar.selectbox(
            "Select Personality",
            [f"{DEFAULT} 🤖", f"{THERAPIST} 🧑‍⚕️", f"{COMEDIAN} 🎭", f"{EXPERT} 💡", f"{CHILD} 🍭"],
        )
        personality = selected_personality.split()[0]

        st.markdown(
            f"""
            <h3 style='text-align: center; letter-spacing: 0.015em;
             font-family: Montserrat, sans-serif; font-weight: 500;'>
            {self.personality_avatars[personality]}
            </h3>""",
            unsafe_allow_html=True,
        )

        self.change_chatbot_type(personality)
        self.avatar[ASSISTANT] = self.personality_avatars[personality]
        self.prompt_template = self.personality_prompts[personality]
        view_chat_history(self.avatar)
        self.get_user_input()


if __name__ == "__main__":
    chatbot = SimpleChatbot()
    chatbot.run()
