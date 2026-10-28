
# Import the unified generation interface.
from app.agents.llm_providers import generate_routed


# Test a simple question expected to route to Gemini.
result = generate_routed(
    question="What is dependency injection?",
    agent="research",
    prompt="Explain dependency injection in one sentence.",
)

# Display the generated answer.
print("\nANSWER:")
print(result["answer"])

# Display the actual provider and routing reason.
print("\nROUTING:")
print(result["routing"])
