
from typing import Literal, TypedDict


# Restrict agent names to our three specialized workers.
AgentName = Literal["research", "code", "summary"]


class AgentState(TypedDict):
    # Original question submitted by the user.
    question: str

    # Agent selected by the supervisor.
    selected_agent: AgentName

    # Final answer returned by the worker.
    answer: str

    # Records which nodes executed, for debugging.
    trace: list[str]
