
import os

from app.rag.retriever import retrieve


from app.rag.vector_store import openai_client

from app.rag.citations import validate_citations


def generate_with_context(question: str) -> dict:
    # STEP 1: RETRIEVAL
    # Find the 5 closest documentation chunks.
    chunks = retrieve(question, k=5)

    # Handle cases where the database contains no chunks.
    if not chunks:
        return {
            "question": question,
            "answer": "I couldn't find relevant documentation.",
            "contexts": [],
            "chunks": [],
            "citation_validation": None,
        }

    # STEP 2: AUGMENTATION
    # Build the context that will be sent to the LLM.
    # Each chunk includes its ID, source, and original text.
    context = "\n\n".join(
        f"[{chunk['id']}]\n"
        f"Source: {chunk['source']}\n"
        f"{chunk['text']}"
        for chunk in chunks
    )

    # Tell the LLM to answer using retrieved evidence only.
    system_prompt = """
You are a developer documentation assistant.

Answer questions using ONLY the supplied documentation.

Rules:
1. Ground factual claims in the retrieved documentation.
2. Cite the exact chunk IDs provided in the context.
3. Each citation must use this format:
   [fastapi/tutorial/request-files.md::0]
4. Never combine multiple chunk IDs into one citation.
5. Never invent chunk IDs or cite documents not provided.
6. If the documentation does not support an answer,
   explicitly say so.
7. Explain technical concepts clearly and concisely.
"""

    # STEP 3: GENERATION
    # Send the retrieved context and question to OpenAI.
    response = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Documentation:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0,
    )

    # Extract the answer from the API response.
    answer = response.choices[0].message.content or ""

    # STEP 4: CITATION VALIDATION
    # Verify that detected citation IDs belong to retrieved chunks.
    validation = validate_citations(answer, chunks)

    # Preserve the raw answer for evaluation, even if citations
    # are invalid. This helps us investigate generation failures.
    return {
        "question": question,
        "answer": answer,
        "contexts": [chunk["text"] for chunk in chunks],
        "chunks": chunks,
        "citation_validation": validation,
    }


def generate_answer(question: str) -> str:
    # Reuse our new function instead of duplicating logic.
    result = generate_with_context(question)

    # Reject responses containing detected invalid citations.
    if (
        result["citation_validation"] is not None
        and not result["citation_validation"]["valid"]
    ):
        return "The generated answer contained invalid citations."

    # Return only the answer for normal application usage.
    return result["answer"]

if __name__ == "__main__":
    # Test generation across three different FastAPI topics.
    test_questions = [
        "How does dependency injection work in FastAPI?",
        "How do I implement OAuth2 authentication in FastAPI?",
        "How can I upload files using FastAPI?",
    ]

    # Run each question through the complete RAG pipeline.
    for question in test_questions:
        print(f"\n{'=' * 60}")
        print(f"QUESTION: {question}\n")

        # Retrieve documentation and generate a cited answer.
        answer = generate_answer(question)

        print(f"ANSWER:\n{answer}")
        print("=" * 60)

