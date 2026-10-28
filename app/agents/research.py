
# Import our RAG pipeline, which returns an answer and its evidence.
from app.rag.generate import generate_with_context

# Import the shared LangGraph state structure.
from app.agents.state import AgentState


def research_agent(state: AgentState) -> dict:
    # Extract the user's question from the shared state.
    question = state["question"]

    try:
        # Run retrieval, augmentation, and generation.
        result = generate_with_context(question)

        # Extract the provider that generated the RAG answer.
        routing = result["routing"]

        # Check whether the answer contains invalid citations.
        validation = result["citation_validation"]

        if validation is not None and not validation["valid"]:
            raise ValueError("Generated answer contains invalid citations")

        # Return the grounded answer and provider details.
        return {
            "answer": result["answer"],
            "routing": routing,
            "trace": state["trace"] + [
                "Research agent executed RAG",
                f"Retrieved {len(result['chunks'])} chunks",
                f"LLM: {routing['provider']} / {routing['model']}",
                f"Routing reason: {routing['reason']}",
            ],
        }

    except Exception as error:
        import traceback
        traceback.print_exc()
        # Handle errors without crashing the entire graph.
        return {
            "answer": "Research agent couldn't complete the request.",
            "trace": state["trace"] + [
                f"Research agent failed: {type(error).__name__}",
            ],
        }
