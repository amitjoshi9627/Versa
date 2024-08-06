import streamlit as st

from chatbot.constants import CHAT_SEPARATOR

st.set_page_config(
    page_title="Streamlit Chatbot",
    page_icon="🤖",
)
st.write("# 💭 Multi Functional Chatbot 🤖")


def main() -> None:
    st.markdown("#### This is your multi-functional chatbot 🤖 with the following usage as of now:")
    st.markdown(
        "1. Chat with Doc 📃: Upload a document and ask questions about it from the chatbot."
    )
    st.markdown(
        "2. Simple Chatbot 🪄: Just have a chit-chat with the Chatbot."
        f"{CHAT_SEPARATOR}Maybe that thing cracks a joke or two? 😄"
    )


if __name__ == "__main__":
    main()
