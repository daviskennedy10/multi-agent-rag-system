
from app.agents.graph import graph

# Request both documentation research and implementation.
question = (
    "Research FastAPI dependency injection "
    "and implement a Python example."
)

# Start a fresh multi-agent workflow.
result = graph.invoke({
    "question": question,
    "selected_agent": "research",
    "answer": "",
    "trace": [],
    "agent_results": {},
})

# Inspect the supervisor's execution plan.
print("\nPLAN:", result["plan"])

# Inspect which agents saved their results.
print("COMPLETED AGENTS:", list(result["agent_results"]))

# Display the full execution history.
print("\nTRACE:")
for event in result["trace"]:
    print("-", event)

# Inspect the Code Agent's output.
print("\nCODE AGENT ANSWER:")
print(result["agent_results"].get("code", "")[:1200])

# Verify the expected agent sequence.
assert result["plan"] == ["research", "code"]

# Verify both agents preserved their findings.
assert "research" in result["agent_results"]
assert "code" in result["agent_results"]

# Verify the supervisor completed the workflow.
assert "Supervisor completed plan" in result["trace"]

print("\nMulti-agent collaboration test passed!")
