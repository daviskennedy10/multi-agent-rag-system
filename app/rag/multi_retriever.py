
from collections import defaultdict

# Import our existing retrieval and query-expansion functions.
from app.rag.retriever import retrieve
from app.rag.query_expansion import expand_query


def multi_retrieve(
    question: str,
    k: int = 5,
    per_query_k: int = 5,
    queries: list[str] | None = None,
) -> list[dict]:
    # Use provided queries for reproducible evaluation.
    # Otherwise, generate new queries using our LLM.
    if queries is None:
        queries = expand_query(question)

    # Always include the original question.
    # Remove duplicates while preserving query order.
    queries = list(dict.fromkeys([question] + queries))

    # Generate alternative queries while preserving the original.
    queries = expand_query(question)

    # Store combined RRF scores for each unique chunk.
    scores = defaultdict(float)

    # Store chunk information using its unique ID.
    chunks_by_id = {}

    # Search Chroma separately for each query.
    for query in queries:
        results = retrieve(query, k=per_query_k)

        # Rank starts at 1 for the best result.
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["id"]

            # RRF rewards chunks appearing high in multiple rankings.
            # 60 is a commonly used rank-smoothing constant.
            scores[chunk_id] += 1 / (60 + rank)

            # Preserve the chunk's text and source metadata.
            chunks_by_id[chunk_id] = chunk

    # Sort unique chunks by their combined RRF score.
    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    # Return only the top-k combined results.
    return [
        {
            **chunks_by_id[chunk_id],
            "rrf_score": scores[chunk_id],
        }
        for chunk_id in ranked_ids[:k]
    ]



if __name__ == "__main__":
    # Use the same question for every evaluation run.
    question = "How do I implement OAuth2 authentication in FastAPI?"

    # Fixed queries let us compare retrieval consistently.
    fixed_queries = [
        "FastAPI OAuth2PasswordBearer configuration",
        "FastAPI token endpoint username password verification",
        "FastAPI OAuth2 JWT token generation and validation",
    ]

    # Retrieve documents using our fixed search queries.
    results = multi_retrieve(
        question,
        k=5,
        queries=fixed_queries,
    )

    # Display the ranked chunks and their RRF scores.
    for index, chunk in enumerate(results, start=1):
        print(f"\nRESULT {index}")
        print(f"ID: {chunk['id']}")
        print(f"RRF score: {chunk['rrf_score']:.5f}")
        print(f"Preview: {chunk['text'][:200]}")

