import os

# ----------------------------------------
# Directory Paths
# ----------------------------------------
TEST_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(TEST_DIR)
DATA_DIR = "tests_data/"
TEST_PDF_FILE_PATH = os.path.join(
    TEST_DIR,
    DATA_DIR,
    "Blue Whale.pdf",
)

# ----------------------------------------
# LLM Model Name
# ----------------------------------------
LLM_MODEL = "Qwen/Qwen2-1.5B-Instruct"

# ----------------------------------------
# Chatbot Role Names
# ----------------------------------------
USER = "user"
ASSISTANT = "assistant"
CHILD = "Child"
COMEDIAN = "Comedian"
DOCBOT = "Docbot"

# ----------------------------------------
# Chat related constants
# ----------------------------------------
CHAT_SEPARATOR = "\n"
