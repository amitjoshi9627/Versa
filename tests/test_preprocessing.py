import pytest

from chatbot.preprocessing import load_data, remove_duplicate, split_item
from tests.constants import TEST_PDF_FILE_PATH


@pytest.fixture
def data() -> list[str]:
    """Provides a sample list of strings for testing.

    Returns:
        list[str]: A list of strings.
    """

    return ["First String", "Second_string", "Third String"]


def test_load_data() -> None:
    """Tests the `load_data` function to ensure it returns a list of strings."""

    assert load_data(TEST_PDF_FILE_PATH)


def test_split_item(data: list[str]) -> None:
    """Tests the `split_item` function to ensure it splits the input strings correctly.

    Args:
        data (list[str]): The list of strings to test.
    """

    expected = ["First", "Stri", "ing", "Secon", "nd_st", "tring", "Third", "Stri", "ing"]
    result = split_item(data, chunk_size=5, chunk_overlap=1)
    assert result == expected


def test_remove_duplicate(data: list[str]) -> None:
    """Tests the `remove_duplicate` function to ensure it correctly removes duplicates.

    Args:
        data (list[str]): The list of strings to test.
    """

    assert len(remove_duplicate(data)) == len(data)
    assert len(remove_duplicate(data + [data[0]])) == len(data)
