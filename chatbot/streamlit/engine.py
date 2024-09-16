import time
from abc import abstractmethod
from typing import Generator, Iterable, Optional

import streamlit as st
from langchain_community.llms.mlx_pipeline import MLXPipeline
from transformers import Pipeline

from chatbot.constants import (
    ASSISTANT,
    CHAT_HISTORY,
    CHATBOT_TYPE,
    CHILD,
    COMEDIAN,
    DEFAULT,
    DEFAULT_BUFFER_LEN,
    DEFAULT_LLM_TEMP,
    DOCBOT,
    EXPERT,
    MACOS,
    MAX_NEW_TOKENS,
    STREAM_SLEEP_TIME,
    THERAPIST,
    USER,
)
from chatbot.login import decrypt_string, login_user
from chatbot.memory import (
    ConversationBufferMemory,
    ConversationMemory,
    ConversationSummaryBufferMemory,
)
from chatbot.prompt import PERSONALITY_PROMPTS, PromptGenerator
from chatbot.streamlit.constants import ACCESS_TOKEN, ACCESS_TOKEN_LINK, KEY, NO, START_VERSA, YES
from chatbot.streamlit.utils import ChatMessage, chat_history_init, load_llm_model
from chatbot.utils import get_os


class StreamlitEngine:
    """StreamlitEngine class for handling chatbot interactions within a Streamlit app.

    Attributes:
        os (str): The operating system.
        llm (AutoModelForCausalLM): The loaded LLM model.
        tokenizer (AutoTokenizer): The tokenizer for the LLM model.
        avatars (dict): A dictionary mapping chatbot types to their corresponding avatar emojis.
        prompt_generator (PromptGenerator): The prompt generator object.
        prompts (dict): A dictionary containing personality prompts for different chatbot types.
    """

    def __init__(self) -> None:
        """Initializes the StreamlitEngine class.

        Loads the LLM model and initializes other attributes if the `START_VERSA` session state is set.
        """

        self.os = get_os()
        self._load_access_token()
        if st.session_state.get(START_VERSA):
            with st.spinner("Loading Model..."):
                self.llm, self.tokenizer = load_llm_model()
            self.avatars = {
                THERAPIST: "🧑‍⚕️",
                EXPERT: "💡",
                CHILD: "🍭",
                COMEDIAN: "🎭",
                DEFAULT: "🤖",
                DOCBOT: "📂",
                USER: "🐼",
            }
            self.prompt_generator = PromptGenerator()
            self.prompts = PERSONALITY_PROMPTS

    def _load_access_token(self) -> None:
        """Loads the Hugging Face access token, either from user input or local storage.

        This method checks if the access token is already stored in the Streamlit session state. If
        not, it prompts the user in the sidebar to either enter a new access token or retrieve one
        from the provided link. It also handles potential errors like invalid access tokens or
        failures when loading from local storage.
        """

        if ACCESS_TOKEN not in st.session_state:
            with st.sidebar:
                has_access_token = st.radio("Do you have access token?", ["Yes", "No"], index=None)
                link_button = st.link_button(
                    "Want to get an Access token?",
                    ACCESS_TOKEN_LINK,
                )
                if has_access_token == YES:
                    link_button.empty()
                    st.text("Hugging Face Access Token (optional)")
                    if access_token := st.text_input("Access Token:", label_visibility="collapsed"):
                        if not access_token.startswith("hf_"):
                            self._display_call_out(
                                "Please enter a valid huggingface access token!",
                                icon="⚠️",
                                call_out_type="warning",
                                wait_time=2.0,
                            )
                        self._login_user(access_token)
                        st.session_state[ACCESS_TOKEN] = access_token
                        st.session_state[START_VERSA] = True
                elif has_access_token == NO:
                    link_button.empty()
                    access_token = self._load_access_token_locally()
                    if access_token:
                        self._login_user(access_token)
                        st.session_state[ACCESS_TOKEN] = access_token
                        st.session_state[START_VERSA] = True
                    else:
                        self._display_call_out(
                            "Error in login! Please Rerun the app.",
                            icon="❌",
                            call_out_type="error",
                            wait_time=2.0,
                        )

    def _load_access_token_locally(
        self,
    ) -> str | None:
        """Attempts to load the access token from Streamlit Secrets.

        This method checks if the access token is stored in Streamlit Secrets under the key `ACCESS_TOKEN`.
        If found, it decrypts the value using the secret key stored under `KEY` and displays a success message.
        Otherwise, it displays an error message.
        """

        if ACCESS_TOKEN in st.secrets:
            access_token = decrypt_string(st.secrets[ACCESS_TOKEN], st.secrets[KEY])
            self._display_call_out(
                "Loaded Access token locally!",
                icon="✅",
                call_out_type="success",
                wait_time=2.0,
            )
        else:
            self._display_call_out("Error Loading Access token!", icon="❌", call_out_type="error", wait_time=2.0)
            access_token = None

        return access_token

    def _login_user(self, access_token: str) -> None:
        """Logs in the user using the provided access token.

        Args:
            access_token (str): The access token to use for login.
        """

        if login_user(access_token=access_token, login_from_dashboard=True):
            self._display_call_out(
                "Hugging Face login successful!",
                icon="✅",
                call_out_type="success",
                wait_time=2.0,
            )
        else:
            self._display_call_out(
                "Invalid token passed! Login not successful.",
                icon="❌",
                call_out_type="error",
                wait_time=2.0,
            )

    @staticmethod
    def _display_call_out(
        message: str,
        icon: Optional[str] = None,
        call_out_type: str = "info",
        wait_time: Optional[float] = None,
    ) -> None:
        """Displays a Streamlit call-out with the specified message, icon, type, and wait time.

        Args:
            message (str): The message to display in the call-out.
            icon (Optional[str]): The icon to display in the call-out. Defaults to None.
            call_out_type (str): The type of call-out to display (e.g., "info", "success", "warning", "error").
              Defaults to "info".
            wait_time (Optional[float]): The time to wait before clearing the call-out (in seconds). Defaults to None.
        """

        call_out = getattr(st, call_out_type)(message, icon=icon)
        if wait_time:
            time.sleep(wait_time)
        call_out.empty()

    def get_pipeline(self, llm_temp: float = DEFAULT_LLM_TEMP) -> Pipeline | MLXPipeline:
        """Gets the appropriate pipeline for the current operating system.

        Args:
            llm_temp (float, optional): The temperature setting for the LLM. Defaults to DEFAULT_LLM_TEMP.

        Returns:
            Union[Pipeline, MLXPipeline]: The pipeline to use for text generation.
        """

        if self.os == MACOS:
            return MLXPipeline(
                model=self.llm,
                tokenizer=self.tokenizer,
                pipeline_kwargs={
                    "temp": llm_temp,
                    "max_tokens": MAX_NEW_TOKENS,
                    "repetition_penalty": 1.1,
                },
            )
        else:
            return Pipeline(
                model=self.llm,
                tokenizer=self.tokenizer,
                task="text-generation",
                do_sample=True,
                temperature=llm_temp,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

    def get_memory(self, memory_type: str = "buffer", buffer_len: int = DEFAULT_BUFFER_LEN) -> ConversationMemory:
        """Returns a ConversationMemory object based on the specified memory type.

        Args:
            memory_type (str): The type of memory to use. Can be 'buffer' or 'summary_buffer'.
              Defaults to 'buffer'.
            buffer_len (int): The length of the buffer to be used directly towards memory.
              Defaults to DEFAULT_BUFFER_LEN.

        Returns:
            ConversationMemory: The ConversationMemory object.
        """

        memory: ConversationMemory
        if memory_type == "buffer":
            memory = ConversationBufferMemory(buffer_len=buffer_len)
        else:
            memory = ConversationSummaryBufferMemory(llm=self.llm, tokenizer=self.tokenizer, buffer_len=buffer_len)
        return memory

    def get_prompt_template(self, chatbot_type: str, with_summary: bool, with_history: bool) -> str:
        """Generates a prompt template for the specified chatbot type, incorporating history and
        summary as needed.

        Args:
            chatbot_type (str): The type of chatbot.
            with_summary (bool): Whether to include a summary of the conversation in the prompt.
            with_history (bool): Whether to include the conversation history in the prompt.

        Returns:
            str: The generated prompt template.
        """

        prompt = self.prompt_generator.generate(
            self.prompts[chatbot_type],
            with_summary=with_summary,
            with_history=with_history,
        )
        return self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
        )

    def get_user_input(self, chatbot_type: str) -> None:
        """Handles user input and generates a response.

        Args:
            chatbot_type (str): The type of chatbot.
        """

        if user_input := st.chat_input("Ask Me Anything!"):
            st.session_state[CHAT_HISTORY].append(ChatMessage(role=USER, message=user_input))
            with st.chat_message(USER, avatar=self.avatars[USER]):
                st.markdown(user_input)

            with st.chat_message(ASSISTANT, avatar=self.avatars[chatbot_type]):
                response = st.write_stream(self.stream_output(self.get_response(chatbot_type, user_input)))
            st.session_state[CHAT_HISTORY].append(ChatMessage(role=ASSISTANT, message=response))

    @staticmethod
    def stream_output(response: str | Iterable[str]) -> Generator[str, None, None]:
        """Streams the given response, yielding chunks of text with a delay.

        Args:
            response (str | Iterable[str]): The response to stream.

        Returns:
            Generator[str, None, None]: A generator that yields chunks of the response.
        """

        for chunk in response:
            yield chunk + ""
            time.sleep(STREAM_SLEEP_TIME)

    @staticmethod
    def change_chatbot_type(chatbot_type: str) -> None:
        """Changes the current chatbot type and initializes the chat history if necessary.

        Args:
            chatbot_type (str): The new chatbot type.
        """

        if st.session_state[CHATBOT_TYPE] != chatbot_type:
            chat_history_init(chatbot_type)

    @abstractmethod
    def get_response(self, chatbot_type: str, query: str) -> str:
        """Implement this method for child class that inherits this class."""

        raise NotImplementedError("Please Implement this method")
