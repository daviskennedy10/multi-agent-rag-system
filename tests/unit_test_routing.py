
# Test our deterministic LLM routing decisions.
from app.agents.llm_router import route_llm


def test_simple_question_routes_to_google():
    # Simple research questions should prefer Gemini.
    result = route_llm("What is FastAPI?", "research")

    assert result["provider"] == "google"
    assert result["reason"]


def test_complex_question_routes_to_openai():
    # Code debugging should prefer OpenAI.
    result = route_llm("Debug this Python function", "code")

    assert result["provider"] == "openai"
    assert result["reason"]


def test_comparison_routes_to_openai():
    # Comparison tasks trigger the complexity heuristic.
    result = route_llm(
        "Compare async and sync performance",
        "research",
    )

    assert result["provider"] == "openai"
