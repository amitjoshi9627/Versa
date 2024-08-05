import pytest

from chatbot.preprocessing import load_data, remove_duplicate, split_item
from tests.constants import TEST_PDF_FILE_PATH


@pytest.fixture
def data() -> list[str]:
    return ["First String", "Second_string", "Third String"]


def test_load_data() -> None:
    assert load_data(TEST_PDF_FILE_PATH)


def test_split_item(data: list[str]) -> None:
    expected = ["First", "Stri", "ing", "Secon", "nd_st", "tring", "Third", "Stri", "ing"]
    result = split_item(data, chunk_size=5, chunk_overlap=1)
    assert result == expected


def test_remove_duplicate(data: list[str]) -> None:
    assert len(remove_duplicate(data)) == len(data)
    assert len(remove_duplicate(data + [data[0]])) == len(data)
