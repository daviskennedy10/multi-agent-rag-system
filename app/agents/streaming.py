
from langgraph.config import get_stream_writer

# Import our existing provider-streaming interface.
from app.agents.llm_providers import stream_routed


def generate_streamed(
    question: str,
    agent: str,
    prompt: str,
) -> dict:
    # Access LangGraph's custom event writer.
    writer = get_stream_writer()

    # Accumulate tokens so the agent can still return
    # a complete answer in its shared state.
    answer_parts = []
    routing = None

    # Generate real tokens using the selected provider.
    for event in stream_routed(question, agent, prompt):

        if event["type"] == "routing":
            # Save and expose the provider decision.
            routing = event["data"]
            writer({
                "type": "routing",
                "data": routing,
            })

        elif event["type"] == "token":
            # Forward the token immediately to LangGraph.
            token = event["data"]
            answer_parts.append(token)

            writer({
                "type": "token",
                "data": token,
            })
        # Update the final provider decision after successful generation.
        elif event["type"] == "complete":
            routing = event["data"]

    # Return the completed answer and routing information.
    return {
        "answer": "".join(answer_parts),
        "routing": routing,
    }
