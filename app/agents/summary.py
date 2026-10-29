
import os


from app.rag.retriever import retrieve

from app.agents.llm_providers import generate_routed

# Import the shared LangGraph state and citation validator.
from app.agents.state import AgentState
from app.rag.citations import validate_citations

# Reuse the streaming interface already used by Research and Code.
from app.agents.streaming import generate_streamed

from app.agents.document_tools import retrieve_related_document


def summary_agent(state: AgentState) -> dict:
    # Read the user's summarization request.
    question = state["question"]

    try:
        # Retrieve documentation relevant to the requested topic.
        chunks = retrieve_related_document(question)

        # Bound the document context to control token usage.
        # Preserve document order while limiting total text.
        MAX_CONTEXT_CHARS = 24000

        selected_chunks = []
        used_chars = 0

        for chunk in chunks:
            # Stop before exceeding our context budget.
            if used_chars + len(chunk["text"]) > MAX_CONTEXT_CHARS:
                break

            selected_chunks.append(chunk)
            used_chars += len(chunk["text"])

        # Use only the selected chunks for generation and citations.
        chunks = selected_chunks

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

        # Stream Gemini or OpenAI tokens through LangGraph.
        # The completed answer is still returned for citation validation.
        result = generate_streamed(
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

        # Preserve the summary for subsequent agents and the supervisor.
        return {
            "answer": answer,
            "routing": routing,

            # Save Summary's findings in shared LangGraph state.
            "agent_results": {
                **state.get("agent_results", {}),
                "summary": answer,
            },

            # Preserve execution history and routing transparency.
            "trace": state["trace"] + [
                "Summary agent executed",
                f"Summary assembled {len(chunks)} document chunks",
                f"LLM: {routing['provider']} / {routing['model']}",
                f"Routing reason: {routing['reason']}",
            ],
        }


    except Exception as error:
        # Record the failure for local debugging.
        print(f"Summary agent error: {type(error).__name__}: {error}")

        # Notify the supervisor that Summary failed.
        return {
            "answer": "Document summarization failed.",

            # Prevent the supervisor from treating this as success.
            "error": f"Summary failed: {type(error).__name__}",

            "trace": state["trace"] + [
                f"Summary agent failed: {type(error).__name__}",
            ],
        }
