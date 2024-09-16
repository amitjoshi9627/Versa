import streamlit as st

st.set_page_config(page_title="Versa: relaxed conversations, real results", page_icon="🤖", layout="wide")
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


def main() -> None:
    """Runs the main Streamlit application.

    Handles the user interface for selecting a chatbot personality and displaying information.
    """

    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <h3 style="font-weight: bold;">Choose Your Vibe</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br><br>", unsafe_allow_html=True)  # Add some space

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Therapist 🧑‍⚕️")
        st.markdown("Need a listening ear? Versa's got you covered.")
    with col2:
        st.markdown("##### Comedian 🎭")
        st.markdown("Craving a laugh? Let's crack some jokes!")
    with col3:
        st.markdown("##### Expert 💡")
        st.markdown("Seeking knowledge? Versa's your personal encyclopedia.")

    st.markdown("<br>", unsafe_allow_html=True)  # Add some space

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("##### Child 🍭")
        st.markdown("Want to unleash your inner kid? Let's play!")
    with col5:
        st.markdown("##### Default 🤖")
        st.markdown("Your Casual chatting partner.")

    st.markdown(
        """
        <div style="text-align: left; margin-top: 50px;">
            <h4>Beyond these personalities, Versa can also:</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)  # Add some space

    st.markdown("##### Dive into Docs 📃")
    st.markdown("Upload a document and ask away anything from **Doc Bot**!")


if __name__ == "__main__":
    main()
