
from app.agents.graph import graph


# Ask a question covered by our indexed FastAPI documentation.
question = "What is FastAPI dependency injection?"

# Initialize the shared LangGraph state.
initial_state = {
    "question": question,
    "selected_agent": "research",
    "answer": "",
    "trace": [],
    "agent_results": {},
}

# Stream both custom LLM events and completed node updates.
for mode, chunk in graph.stream(
    initial_state,
    stream_mode=["custom", "updates"],
):
    if mode == "custom":
        if chunk["type"] == "routing":
            # Display the selected provider.
            print("\nROUTING:", chunk["data"])

        elif chunk["type"] == "token":
            # Print genuine generated text immediately.
            print(chunk["data"], end="", flush=True)

    elif mode == "updates":
        # Show which LangGraph node completed.
        print("\nNODE COMPLETED:", list(chunk.keys()))
