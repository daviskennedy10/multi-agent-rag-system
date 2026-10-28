
import os

# Import our Chroma retriever and OpenAI client.
from app.rag.retriever import retrieve
from app.rag.vector_store import openai_client

# Import the shared LangGraph state and citation validator.
from app.agents.state import AgentState
from app.rag.citations import validate_citations


def summary_agent(state: AgentState) -> dict:
    # Read the user's summarization request.
    question = state["question"]

    try:
        # Retrieve documentation relevant to the requested topic.
        chunks = retrieve(question, k=5)

        # Handle cases where no documentation is available.
        if not chunks:
            return {
                "answer": "No relevant documentation was found.",
                "trace": state["trace"] + [
                    "Summary agent found no documentation"
                ],
            }

        # Combine retrieved passages with their source IDs.
        context = "\n\n".join(
            f"[{chunk['id']}]\n{chunk['text']}"
            for chunk in chunks
        )

        # Give this agent a distinct summarization role.
        system_prompt = """
You are a specialized Technical Documentation Summary Agent.

Your responsibilities:
1. Summarize the supplied documentation concisely.
2. Preserve important technical concepts.
3. Use ONLY information supported by the retrieved text.
4. Cite the exact chunk IDs provided in the context.
5. Each citation must contain exactly ONE chunk ID.
6. Never combine multiple chunk numbers in one citation.
7. Never invent citations or source information.

Correct:
[fastapi/tutorial/security/first-steps.md::7]
[fastapi/tutorial/security/first-steps.md::12]

Incorrect:
[fastapi/tutorial/security/first-steps.md::7,12]

If multiple chunks support a claim, cite them separately.
"""

        # Ask the LLM to summarize the retrieved documentation.
        response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Documentation:\n{context}\n\n"
                        f"Summarization request: {question}"
                    ),
                },
            ],
            temperature=0,
        )

        # Extract the generated summary.
        answer = response.choices[0].message.content or ""

        # Verify that detected citations refer to retrieved chunks.
        validation = validate_citations(answer, chunks)

        # Temporarily inspect the generated answer and citation results.
        print("\nDEBUG — SUMMARY ANSWER:")
        print(answer)

        # Show which citations passed or failed validation.
        print("\nDEBUG — CITATION VALIDATION:")
        print(validation)

        if not validation["valid"]:
            raise ValueError("Summary contains invalid citations")

        # Return the summary and execution trace.
        return {
            "answer": answer,
            "trace": state["trace"] + [
                "Summary agent executed",
                f"Summarized {len(chunks)} retrieved chunks",
            ],
        }

    except Exception as error:
        # Display the actual error while debugging.
        print(f"Summary agent error: {type(error).__name__}: {error}")

        # Return a safe failure response without crashing LangGraph.
        return {
            "answer": "Document summarization failed.",
            "trace": state["trace"] + [
                f"Summary agent failed: {type(error).__name__}",
            ],
        }
