
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

# Import our streaming helper.
from app.agents.streaming import generate_streamed


class TestState(TypedDict):
    # Input question and final generated answer.
    question: str
    answer: str


def streaming_node(state: TestState) -> dict:
    # Stream the response using our existing provider router.
    result = generate_streamed(
        question=state["question"],
        agent="research",
        prompt="Explain dependency injection in two sentences.",
    )

    # Save the final answer in LangGraph state.
    return {"answer": result["answer"]}


# Build a minimal graph for testing custom events.
builder = StateGraph(TestState)
builder.add_node("generate", streaming_node)
builder.add_edge(START, "generate")
builder.add_edge("generate", END)

graph = builder.compile()


# Execute the graph and receive custom streaming events.
for mode, chunk in graph.stream(
    {"question": "What is dependency injection?", "answer": ""},
    stream_mode=["custom", "updates"],
):
    if mode == "custom":
        # Display tokens as the model generates them.
        if chunk["type"] == "token":
            print(chunk["data"], end="", flush=True)

        elif chunk["type"] == "routing":
            print("\nROUTING:", chunk["data"])

    elif mode == "updates":
        # Confirm the graph node completed.
        print("\nNODE UPDATE:", list(chunk.keys()))
