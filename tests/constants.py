import os

TEST_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(TEST_DIR)
DATA_DIR = "tests_data/"
TEST_PDF_FILE_PATH = os.path.join(
    TEST_DIR,
    DATA_DIR,
    "Blue Whale.pdf",
)

USER = "user"
ASSISTANT = "assistant"
CHILD = "Child"
COMEDIAN = "Comedian"
DOCBOT = "Docbot"

CHAT_SEPARATOR = "\n"
