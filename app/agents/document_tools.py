
from app.rag.vector_store import collection


def retrieve_document(source: str) -> list[dict]:
    """Retrieve all indexed chunks belonging to one document."""

    # Filter Chroma by the original documentation source.
    results = collection.get(
        where={"source": source},
        include=["documents", "metadatas"],
    )

    # Reconstruct chunks with their original identifiers.
    chunks = [
        {
            "id": chunk_id,
            "text": text,
            "source": metadata["source"],
        }
        for chunk_id, text, metadata in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        )
    ]

    # Restore document order using the numeric chunk suffix.
    chunks.sort(
        key=lambda chunk: int(chunk["id"].rsplit("::", 1)[1])
    )

    return chunks


def retrieve_related_document(question: str) -> list[dict]:
    """Find the most relevant document and assemble its chunks."""

    # Import here to avoid unnecessary initialization cycles.
    from app.rag.retriever import retrieve

    # Identify the document most relevant to the question.
    matches = retrieve(question, k=1)

    if not matches:
        return []

    # Retrieve all chunks from that document.
    return retrieve_document(matches[0]["source"])
