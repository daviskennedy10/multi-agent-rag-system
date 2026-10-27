
import os

from app.rag.retriever import retrieve


from app.rag.vector_store import openai_client


def generate_answer(question: str) -> str:
    # STEP 1: RETRIEVAL
    # Search Chroma for the 5 most relevant document chunks.
    chunks = retrieve(question, k=5)

    # If our database is empty, we cannot generate
    # an answer grounded in documentation.
    if not chunks:
        return "I couldnt find relevant information in the available documentation."

    # STEP 2: AUGMENTATION
    # Combine retrieved chunks into one context string.
    # Include chunk IDs and filenames so the LLM can cite them.
    context = "\n\n".join(
        f"[{chunk['id']}]\n"
        f"Source: {chunk['source']}\n"
        f"{chunk['text']}"
        for chunk in chunks
    )

    # Define the rules that control the LLM's behavior.
    # These instructions help reduce hallucinations.
    system_prompt = """
You are a developer documentation assistant.

Answer questions using ONLY the supplied documentation.

Rules:
1. Ground every factual claim in the provided context.
2. Cite the source chunk IDs, e.g. [fastapi_basics_0].
3. If the documentation does not contain the answer,
   explicitly say so.
4. Never invent citations.
5. Explain technical concepts clearly and concisely.
"""

    # STEP 3: GENERATION
    # Send the instructions, retrieved documentation,
    # and user's question to the OpenAI model.
    response = openai_client.chat.completions.create(
        # Use OPENAI_MODEL from .env if available.
        # Otherwise, use gpt-4.1-mini as our prototype model.
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),

        messages=[
            # System message defines the assistant's behavior.
            {"role": "system", "content": system_prompt},

            # User message contains the retrieved documentation
            # alongside the actual question.
            {
                "role": "user",
                "content": (
                    f"Documentation:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],

        # Temperature 0 requests less randomness.
        # It does not guarantee a correct or identical answer.
        temperature=0,
    )

    # Extract the generated text from the API response.
    # Return an empty string if the response has no text.
    return response.choices[0].message.content or ""



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

