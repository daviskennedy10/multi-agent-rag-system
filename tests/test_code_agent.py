
from app.agents.code import code_agent


# Example Python code containing a division-by-zero bug.
sample_code = """
def divide(a, b):
    return a / b

print(divide(10, 0))
"""

# Simulate the state normally provided by LangGraph.
state = {
    "question": sample_code,
    "selected_agent": "code",
    "answer": "",
    "trace": [],
}

# Execute the Code Agent directly.
result = code_agent(state)

# Display the generated analysis and trace.
print("ANSWER:\n", result["answer"])
print("\nTRACE:", result["trace"])
