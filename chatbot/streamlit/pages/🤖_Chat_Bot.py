import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from chatbot.constants import (
    ASSISTANT,
    CHAT_HISTORY,
    CHATBOT_TYPE,
    CHILD,
    COMEDIAN,
    CREATIVE_LLM_TEMP,
    DEFAULT,
    EXPERT,
    THERAPIST,
)
from chatbot.streamlit.engine import StreamlitEngine
from chatbot.streamlit.utils import (
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


class StreamlitChatBotEngine(StreamlitEngine):
    def __init__(self) -> None:
        super().__init__()

    def get_response(self, chatbot_type: str, query: str) -> str:
        pipeline = self.get_pipeline(llm_temp=CREATIVE_LLM_TEMP)

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
                "history": chat_history,
            }
        )

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
            {self.avatars[personality]}
            </h3>""",
            unsafe_allow_html=True,
        )

        self.avatars[ASSISTANT] = self.avatars[personality]
        self.change_chatbot_type(personality)
        view_chat_history(self.avatars)
        self.get_user_input(personality)


if __name__ == "__main__":
    chatbot = StreamlitChatBotEngine()
    chatbot.run()
