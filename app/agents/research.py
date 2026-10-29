
# Import our RAG pipeline, which returns an answer and its evidence.
from app.rag.generate import generate_with_context

# Import the shared LangGraph state structure.
from app.agents.state import AgentState

# Run asynchronous MCP tools from our synchronous LangGraph node.
import asyncio

# Shared Tavily MCP search function.
from app.agents.web_tools import search_web


def research_agent(state: AgentState) -> dict:
    # Extract the user's question from the shared state.
    question = state["question"]

    # Search the live web when the user requests current information.
    live_keywords = ["latest", "current", "recent", "today", "2026"]

    needs_live_search = any(
        keyword in question.lower()
        for keyword in live_keywords
    )

    # Keep the web results separate from our static RAG evidence.
    web_context = ""

    if needs_live_search:
        # Invoke the shared Tavily MCP search tool.
        web_context = asyncio.run(search_web(question))

    try:
        # Generate real model tokens through LangGraph.
        # The completed answer remains available for citation validation.
        result = generate_with_context(
            question,
            web_context=web_context,
            stream=True,
        )

        # Extract the provider that generated the RAG answer.
        routing = result["routing"]

        # Check whether the answer contains invalid citations.
        validation = result["citation_validation"]

        # Inspect citation validation after streaming completes.
        # This helps identify malformed or invented citations.
        print("RESEARCH CITATION VALIDATION:", validation, flush=True)

        # Check whether the generated citations passed validation.
        citations_valid = (
            validation is not None
            and validation["valid"]
        )

        # Record citation problems instead of hiding the completed response.
        citation_status = (
            "Citation IDs validated"
            if citations_valid
            else "WARNING: Citation validation failed"
        )

        # Return the grounded answer and provider details.
        return {
            "answer": result["answer"],
            "routing": routing,
            "trace": state["trace"] + [
                "Research agent executed RAG",
                f"Retrieved {len(result['chunks'])} chunks",

                # Record whether Research actually invoked Tavily.
                *(
                    ["Research invoked MCP tool: tavily_search"]
                    if needs_live_search
                    else []
                ),

                f"LLM: {routing['provider']} / {routing['model']}",
                f"Routing reason: {routing['reason']}",
                f"citation status: {citation_status}",
            ],
            "agent_results": {
                **state.get("agent_results", {}),
                "research": result["answer"],
            },
            
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
