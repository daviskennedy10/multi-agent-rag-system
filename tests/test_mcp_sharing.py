
from app.agents.graph import graph


# Two different requests requiring live information.
test_cases = [
    (
        "Research",
        "What are the latest FastAPI release changes?",
    ),
    (
        "Code",
        "Analyze this code using the latest FastAPI API:\n"
        "from fastapi import FastAPI\n"
        "app = FastAPI()",
    ),
]


for name, question in test_cases:
    # Start a fresh LangGraph workflow for each request.
    result = graph.invoke({
        "question": question,
        "selected_agent": "research",
        "answer": "",
        "trace": [],
        "agent_results": {},
    })

    # Display which agent executed and its tool trace.
    print(f"\n{name} TEST")
    print("Plan:", result["plan"])
    print("Trace:")

    for event in result["trace"]:
        print("-", event)

    # Confirm that the expected agent invoked Tavily.
    expected = f"{name} invoked MCP tool: tavily_search"

    assert expected in result["trace"], (
        f"{name} did not invoke Tavily MCP"
    )

    print(f"{name} MCP sharing test passed!")
