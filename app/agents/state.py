
from typing import Literal, NotRequired, TypedDict


# Restrict worker names to our three specialized agents.
AgentName = Literal["research", "code", "summary"]


class AgentState(TypedDict):
    # Original user request.
    question: str

    # The next agent selected by the supervisor.
    selected_agent: AgentName

    # Most recent agent's response.
    answer: str

    # History of workflow execution.
    trace: list[str]

    # Ordered list of agents that should execute.
    plan: NotRequired[list[AgentName]]

    # Tracks which planned agent executes next.
    step: NotRequired[int]

    # Stores each agent's output for later coordination.
    agent_results: NotRequired[dict[str, str]]

    # Actual provider and model used by the latest agent.
    routing: NotRequired[dict]
