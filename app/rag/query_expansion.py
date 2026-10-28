
import os

from app.rag.vector_store import openai_client


def expand_query(question: str) -> list[str]:
    # Ask a lightweight LLM to generate additional search queries.
    response = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate exactly 3 distinct search queries "
                    "that would help retrieve documentation needed "
                    "to answer the user's question. "
                    "Return only the queries, one per line."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0,
    )

    # Split the generated text into individual queries.
    generated = response.choices[0].message.content or ""

    # Remove blank lines and common numbered-list prefixes.
    queries = [
        line.strip().lstrip("123.-) ")
        for line in generated.splitlines()
        if line.strip()
    ]

    # Include the original question to preserve its intent.
    return [question] + queries[:3]


if __name__ == "__main__":
    # Test query expansion on our OAuth2 question.
    question = "How do I implement OAuth2 authentication in FastAPI?"

    for query in expand_query(question):
        print("-", query)
