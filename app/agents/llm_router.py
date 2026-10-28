
import os
from typing import TypedDict


class RoutingDecision(TypedDict):
    # Which LLM provider should handle the request?
    provider: str

    # Exact model identifier to call.
    model: str

    # Explanation for why this provider was selected.
    reason: str


def route_llm(question: str, agent: str) -> RoutingDecision:
    # Identify signals that suggest a more complex task.
    complexity_signals = [
        "debug",
        "analyze",
        "refactor",
        "optimize",
        "compare",
        "architecture",
        "tradeoff",
        "security vulnerability",
    ]

    # Count distinct complexity signals in the question.
    question_lower = question.lower()

    complexity_score = sum(
        signal in question_lower
        for signal in complexity_signals
    )

    # Code analysis may require deeper reasoning.
    # This is a heuristic, not a measured capability score.
    if agent == "code":
        complexity_score += 1

    # Route more complex tasks to OpenAI.
    if complexity_score >= 1:
        return {
            "provider": "openai",
            "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "reason": (
                f"Complexity score {complexity_score}; "
                "prioritizing reasoning capability."
            ),
        }

    # Route simpler tasks to Gemini.
    return {
        "provider": "google",
        "model": os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        "reason": (
            f"Complexity score {complexity_score}; "
            "prioritizing cost-efficient generation."
        ),
    }


if __name__ == "__main__":
    # Test routing decisions without calling either provider.
    tests = [
        ("What is FastAPI?", "research"),
        ("Debug this Python function", "code"),
        ("Summarize dependency injection", "summary"),
        ("Compare async and sync performance", "research"),
    ]

    for question, agent in tests:
        # Display the selected provider and its rationale.
        decision = route_llm(question, agent)

        print(f"\nQuestion: {question}")
        print(f"Agent: {agent}")
        print(f"Decision: {decision}")
