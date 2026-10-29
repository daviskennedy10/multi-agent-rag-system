
from app.agents.graph import supervisor, route_agent
from app.rag.citations import validate_citations


def test_supervisor_stops_after_failure():
    # Simulate Research failing before Code executes.
    state = {
        "question": "Research FastAPI and implement an example",
        "plan": ["research", "code"],
        "step": 0,
        "selected_agent": "research",
        "answer": "Research failed",
        "error": "Research failed: ServerError",
        "trace": ["Research agent failed: ServerError"],
        "agent_results": {},
    }

    # Apply the supervisor's state update.
    updated = {**state, **supervisor(state)}

    # Verify the graph stops instead of delegating to Code.
    assert route_agent(updated) == "end"
    assert "Supervisor stopped plan after agent failure" in updated["trace"]
    assert "Supervisor delegated to code" not in updated["trace"]


def test_valid_citation():
    # A citation must reference a retrieved chunk.
    chunks = [
        {
            "id": "fastapi/tutorial/dependencies/index.md::0",
            "text": "FastAPI supports dependency injection.",
        }
    ]

    answer = (
        "FastAPI supports dependencies "
        "[fastapi/tutorial/dependencies/index.md::0]"
    )

    result = validate_citations(answer, chunks)

    assert result["valid"] is True


def test_invalid_citation():
    # Reject citations referring to chunks not retrieved.
    chunks = [
        {
            "id": "fastapi/tutorial/dependencies/index.md::0",
            "text": "FastAPI supports dependency injection.",
        }
    ]

    answer = (
        "FastAPI supports dependencies "
        "[fastapi/tutorial/dependencies/index.md::999]"
    )

    result = validate_citations(answer, chunks)

    assert result["valid"] is False


def test_routing_ends_completed_plan():
    # The graph must terminate when all planned agents finish.
    state = {
        "plan": ["research", "code"],
        "step": 2,
        "selected_agent": "code",
    }

    assert route_agent(state) == "end"
