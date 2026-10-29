
# Test our deterministic fallback planner without LLM calls.
from app.agents.planner import create_plan


def test_research_request():
    # General documentation questions require Research.
    assert create_plan("Explain FastAPI") == ["research"]


def test_code_request():
    # Debugging requests require Code.
    assert create_plan("Debug this function") == ["code"]


def test_multi_agent_request():
    # Research followed by implementation needs both agents.
    assert create_plan(
        "Research FastAPI and implement an example"
    ) == ["research", "code"]
