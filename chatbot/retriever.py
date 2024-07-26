from langchain_community.vectorstores import (
    FAISS,
)


def retrieve_docs(
    query: str,
    knowledge_index: FAISS,
    num_docs: int = 5,
) -> list[str]:
    relevant_docs = [
        doc.page_content
        for doc in knowledge_index.similarity_search(
            query=query,
            k=num_docs,
        )
    ]

    return relevant_docs
