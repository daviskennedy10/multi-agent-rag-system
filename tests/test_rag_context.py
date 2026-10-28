
from app.rag.generate import generate_with_context


# Ask a question covered by our FastAPI documentation.
question = "How does dependency injection work in FastAPI?"

# Run the complete retrieval and generation pipeline.
result = generate_with_context(question)

# Check that the required evaluation fields exist.
assert result["question"] == question
assert isinstance(result["answer"], str)
assert isinstance(result["contexts"], list)
assert isinstance(result["chunks"], list)

# Verify that the retrieved context is available.
assert len(result["contexts"]) > 0

# Display a compact summary of the result.
print(f"Question: {result['question']}")
print(f"Answer: {result['answer'][:200]}...")
print(f"Retrieved chunks: {len(result['chunks'])}")
print(f"Context count: {len(result['contexts'])}")
print(f"Citation validation: {result['citation_validation']}")

print("\nRAG context test passed!")
