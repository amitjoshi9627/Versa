from langchain_community.vectorstores import (
    FAISS,
)


def retrieve_docs(
    query: str,
    knowledge_index: FAISS,
    num_docs: int = 5,
) -> list[str]:
    """Retrieves relevant documents from the given knowledge index using similarity search.

    Args:
        query (str): The query to search for.
        knowledge_index (FAISS): The FAISS index containing the document embeddings.
        num_docs (int, optional): The number of documents to retrieve. Defaults to 5.

    Returns:
        list[str]: A list of relevant document texts.
    """

    relevant_docs = [
        doc.page_content
        for doc in knowledge_index.similarity_search(
            query=query,
            k=num_docs,
        )
    ]

    return relevant_docs
