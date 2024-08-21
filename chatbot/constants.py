import os

DATA_DIR = "data/"
MODEL_DIR = "saved_models/"
PDF_FILE_PATH = os.path.join(
    DATA_DIR,
    "sample_data/" "Blue Whale.pdf",
)

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
UPLOAD_REPO = "mlx-community/My-Mistral-7B-Instruct-v0.3-4bit"

# less than 1 for more deterministic or greater than 1 for more creative
CREATIVE_LLM_TEMP = 1.1
DETERMINISTIC_LLM_TEMP = 0.2
DEFAULT_LLM_TEMP = 1.0
MAX_NEW_TOKENS = 512  # Length of output token generated


SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_PATH = os.path.join(
    MODEL_DIR,
    "Sentence Transformer",
)
VECTOR_DB_PATH = os.path.join(
    "Database/",
    "vector_db",
)
VECTOR_DB_INDEX = "vectorDBIndex"

SEPARATORS = [
    "\n\n",
    "\n",
    " ",
    "",
]
CHAT_SEPARATOR = "\n"

CHUNK_SIZE = 512
CHUNK_OVERLAP = CHUNK_SIZE // 10
DEFAULT_BUFFER_LEN = 12
STREAM_SLEEP_TIME = 0.005

# constant names
CHAT_HISTORY = "chat_history"
DATABASE = "Database"
ASSISTANT = "assistant"
USER = "user"
HUMAN = "human"
MESSAGE = "message"
ROLE = "role"
CONTENT = "content"
CHATBOT_TYPE = "chatbot_type"

# platform based constants
LINUX = "Linux"
WINDOWS = "Windows"
MACOS = "Darwin"

# GPU related constants
CUDA = "cuda"
MPS = "mps"
CPU = "cpu"

# personality based constants
THERAPIST = "Therapist"
COMEDIAN = "Comedian"
CHILD = "Child"
EXPERT = "Expert"
DEFAULT = "Default"
DOCBOT = "Docbot"

CHATBOT_TYPE_LIST = [
    THERAPIST,
    COMEDIAN,
    CHILD,
    EXPERT,
    DEFAULT,
]


# prompt based constants
HISTORY = "history"
SUMMARY = "summary"
QUERY = "query"
GENERATED_TEXT = "generated_text"
