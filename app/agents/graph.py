
from langgraph.graph import StateGraph, START, END

# Import our shared state definition.
from app.agents.state import AgentState

from app.agents.research import research_agent


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

def code_agent(state: AgentState) -> dict:
    # Placeholder: real code-analysis tools will be added later.
    return {
        "answer": "Code agent received the question.",
        "trace": state["trace"] + ["Code agent executed"],
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
    # Ask a documentation question that should route to Research.
    question = "How does dependency injection work in FastAPI?"

    # Execute the complete LangGraph workflow.
    result = graph.invoke({
        "question": question,
        "selected_agent": "research",
        "answer": "",
        "trace": [],
    })

    # Inspect which agent executed and what it returned.
    print(f"\nSelected agent: {result['selected_agent']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nTrace: {result['trace']}")
