import os.path

import torch
from langchain_community.vectorstores import (
    FAISS,
)
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
)
from langchain_core.documents import (
    Document,
)
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from chatbot.constants import SENTENCE_TRANSFORMER_MODEL, VECTOR_DB_INDEX, VECTOR_DB_PATH


def get_vector_database(
    chunks: list[Document] | list[str] | None = None,
    save_database: bool = True,
) -> FAISS:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    embeddings = HuggingFaceEmbeddings(
        model_name=SENTENCE_TRANSFORMER_MODEL,
        multi_process=True,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    if chunks:
        if isinstance(
            chunks[0],
            Document,
        ):
            vector_database = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings,
                distance_strategy=DistanceStrategy.COSINE,
            )
        elif isinstance(
            chunks[0],
            str,
        ):
            vector_database = FAISS.from_texts(
                texts=chunks,
                embedding=embeddings,
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            raise TypeError(f"Type {type(chunks[0])} not supported currently.")

        if save_database:
            vector_database.save_local(
                folder_path=VECTOR_DB_PATH,
                index_name=VECTOR_DB_INDEX,
            )

    elif os.path.exists(VECTOR_DB_PATH):
        vector_database = FAISS.load_local(
            folder_path=VECTOR_DB_PATH,
            index_name=VECTOR_DB_INDEX,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        raise ValueError("Document not provided to create database.")
    return vector_database
