from app.rag.vector_store import collection, embed_text

# max_distance defines the largest acceptable vector distance.
# 1.0 is an initial experimental threshold, not a universal cutoff.
def retrieve(query: str, k: int = 5, max_distance: float | None = None,) -> list[dict]:
    if not query.strip():
        raise ValueError("Query cannot be empty")

    if k <= 0:
        raise ValueError("k must be positive")

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    for i, chunk_id in enumerate(results["ids"][0]):
        # Read the distance between the question and this chunk.
        distance = results["distances"][0][i]

        # Skip chunks that exceed our configured distance threshold.
        if max_distance is not None and distance > max_distance:
            continue
        chunks.append({
            "id": chunk_id,
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })

    return chunks


if __name__ == "__main__":
    query = "How does FastAPI support asynchronous functions?"

    results = retrieve(query)

    for result in results:
        print(f"\nID: {result['id']}")
        print(f"Source: {result['source']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text: {result['text'][:200]}")