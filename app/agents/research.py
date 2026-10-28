
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

        # Check whether the answer contains invalid citations.
        validation = result["citation_validation"]

        if validation is not None and not validation["valid"]:
            raise ValueError("Generated answer contains invalid citations")

        # Return state updates to LangGraph.
        return {
            "answer": result["answer"],
            "trace": state["trace"] + [
                "Research agent executed RAG",
                f"Retrieved {len(result['chunks'])} chunks",
            ],
        }

    except Exception as error:
        # Handle errors without crashing the entire graph.
        return {
            "answer": "Research agent couldn't complete the request.",
            "trace": state["trace"] + [
                f"Research agent failed: {type(error).__name__}",
            ],
        }
