
from app.agents.state import AgentName


def create_plan(question: str) -> list[AgentName]:
    # Normalize the question for basic intent detection.
    text = question.lower()

    # Detect requests involving code or implementation.
    needs_code = any(
        word in text
        for word in ["code", "implement", "debug", "function"]
    )

    # Detect requests asking for a summary.
    needs_summary = any(
        word in text
        for word in ["summarize", "summary"]
    )

    # Detect requests requiring documentation research.
    needs_research = any(
        word in text
        for word in ["research", "explain", "documentation"]
    )

    # Combine research and code when both are needed.
    if needs_code and needs_research:
        return ["research", "code"]

    # Otherwise, select the most relevant specialist.
    if needs_code:
        return ["code"]

    if needs_summary:
        return ["summary"]

    return ["research"]


if __name__ == "__main__":
    # Test whether different requests produce different plans.
    questions = [
        "What is FastAPI?",
        "Debug this Python function",
        "Summarize FastAPI authentication",
        "Research FastAPI dependencies and implement an example",
    ]

    for question in questions:
        # Display the planned agent sequence.
        print(question, "->", create_plan(question))
