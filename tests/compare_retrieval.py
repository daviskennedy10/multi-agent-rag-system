
from app.rag.generate import generate_with_context


# Use the same question for both retrieval strategies.
question = "How do I implement OAuth2 authentication in FastAPI?"


for use_multi_query in [False, True]:
    # Identify which retrieval strategy we're testing.
    strategy = "Multi-query" if use_multi_query else "Original"

    print(f"\n{'=' * 60}")
    print(f"STRATEGY: {strategy}")

    # Generate an answer using the selected retrieval strategy.
    result = generate_with_context(
        question,
        use_multi_query=use_multi_query,
    )

    # Display the generated answer.
    print(f"\nANSWER:\n{result['answer']}")

    # Display the documentation chunks used.
    print("\nRETRIEVED SOURCES:")
    for chunk in result["chunks"]:
        print("-", chunk["id"])

    # Check whether citations reference retrieved chunks.
    print("\nCITATION VALIDATION:")
    print(result["citation_validation"])
