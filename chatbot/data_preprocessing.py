from typing import (
    Any,
)

from langchain_core.documents import (
    Document,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from PyPDF2 import (
    PdfReader,
)
from streamlit.runtime.uploaded_file_manager import (
    UploadedFile,
)

from chatbot.constants import CHUNK_OVERLAP, CHUNK_SIZE, SEPARATORS


def load_data(
    file_path: str | UploadedFile,
) -> list[str]:
    doc = []
    pdf_reader = PdfReader(file_path)
    for pages in pdf_reader.pages:
        doc += [pages.extract_text()]
    return doc


def split_item(
    item_list: list[Document] | list[str],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    **kwargs: Any,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        strip_whitespace=True,
        **kwargs,
    )

    item_processed = []
    for item in item_list:
        if isinstance(
            item,
            str,
        ):
            item_processed += splitter.split_text(item)
        else:
            item_processed += splitter.split_documents([item])

    return item_processed


def remove_duplicate(
    item_list: list[Document] | list[str],
) -> list[Document] | list[str]:
    item_flag = {}
    unique_item = []
    for item in item_list:
        if hasattr(
            item,
            "page_content",
        ):
            item = item.page_content
        if item not in item_flag:
            item_flag[item] = True
            unique_item.append(item)

    return unique_item
