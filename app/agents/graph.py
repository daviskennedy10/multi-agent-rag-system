
from langgraph.graph import StateGraph, START, END

# Import our shared state definition.
from app.agents.state import AgentState

from app.agents.research import research_agent

from app.agents.code import code_agent


def supervisor(state: AgentState) -> dict:
    # Temporary routing logic to verify graph execution.
    # We'll replace this with intelligent LLM routing.
    question = state["question"].lower()

    if "code" in question or "function" in question:
        selected = "code"
    elif "summarize" in question or "summary" in question:
        selected = "summary"
    else:
        selected = "research"

    # Update the state with the selected agent and trace.
    return {
        "selected_agent": selected,
        "trace": state["trace"] + [f"Supervisor selected {selected}"],
    }


def summary_agent(state: AgentState) -> dict:
    # Placeholder: document summarization comes later.
    return {
        "answer": "Summary agent received the question.",
        "trace": state["trace"] + ["Summary agent executed"],
    }


def route_agent(state: AgentState) -> str:
    # LangGraph uses this value to select the next node.
    return state["selected_agent"]


# Create a graph whose shared state follows AgentState.
builder = StateGraph(AgentState)

# Register the supervisor and worker nodes.
builder.add_node("supervisor", supervisor)
builder.add_node("research", research_agent)
builder.add_node("code", code_agent)
builder.add_node("summary", summary_agent)

# Start execution at the supervisor.
builder.add_edge(START, "supervisor")

# Dynamically select the next node from the supervisor's state.
builder.add_conditional_edges(
    "supervisor",
    route_agent,
    {
        "research": "research",
        "code": "code",
        "summary": "summary",
    },
)

# End after the selected worker executes.
for agent in ["research", "code", "summary"]:
    builder.add_edge(agent, END)

# Compile the graph into an executable workflow.
graph = builder.compile()


if __name__ == "__main__":
    # Provide Python code containing a division-by-zero bug.
    question = """
Analyze this Python code:

def divide(a, b):
    return a / b

print(divide(10, 0))
"""

    # Execute the complete supervisor workflow.
    result = graph.invoke({
        "question": question,
        "selected_agent": "research",
        "answer": "",
        "trace": [],
    })

    # Display which agent the supervisor selected.
    print(f"\nSelected agent: {result['selected_agent']}")

    # Display the agent's analysis.
    print(f"\nAnswer:\n{result['answer']}")

    # Display the nodes that executed.
    print(f"\nTrace: {result['trace']}")

