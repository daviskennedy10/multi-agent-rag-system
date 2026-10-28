
import os


from app.rag.retriever import retrieve

from app.agents.llm_providers import generate_routed

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

        
        # Combine the summarization instructions and retrieved context.
        prompt = (
            f"{system_prompt}\n\n"
            f"Documentation:\n{context}\n\n"
            f"Summarization request: {question}"
        )

        # Select a provider based on task complexity.
        result = generate_routed(
            question=question,
            agent="summary",
            prompt=prompt,
        )

        # Extract the answer and actual provider decision.
        answer = result["answer"]
        routing = result["routing"]


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

        # Return the summary and provider-routing information.
        return {
            "answer": answer,
            "routing": routing,
            "trace": state["trace"] + [
                "Summary agent executed",
                f"Summarized {len(chunks)} retrieved chunks",
                f"LLM: {routing['provider']} / {routing['model']}",
                f"Routing reason: {routing['reason']}",
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
