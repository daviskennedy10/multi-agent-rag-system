
import os
from typing import Literal

from pydantic import BaseModel, Field

# Reuse our existing OpenAI client.
from app.agents.llm_providers import openai_client


class AgentPlan(BaseModel):
    # Ordered list of agents needed to complete the request.
    agents: list[
        Literal["research", "code", "summary"]
    ] = Field(min_length=1, max_length=3)

    # Explain why the selected agents are necessary.
    reason: str


def create_llm_plan(question: str) -> AgentPlan:
    # Ask the LLM to produce a validated structured plan.
    response = openai_client.beta.chat.completions.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {
                "role": "system",
                "content": """
You are the supervisor planner for a developer documentation assistant.

Available agents:
- research: Retrieves documentation and answers technical questions.
- code: Analyzes Python code, finds bugs, and creates examples.
- summary: Produces concise summaries of technical documentation.

Rules:
1. Select the minimum number of agents needed.
2. Use research for documentation questions.
3. Use code for debugging, code analysis, or implementation examples.
4. Use summary for summarization requests.
5. For research followed by implementation, use research then code.
6. For research followed by summarization, use research then summary.
7. Never select more than three agents.
8. Return a concise explanation of your decision.
""",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        response_format=AgentPlan,
    )

    # Return the validated plan.
    plan = response.choices[0].message.parsed

    if plan is None:
        raise ValueError("LLM did not return a valid agent plan")

    return plan


if __name__ == "__main__":
    # Test planning without executing any worker agents.
    questions = [
        "What is FastAPI dependency injection?",
        "Debug this Python function.",
        "Summarize FastAPI OAuth2 authentication.",
        "Research dependency injection and implement an example.",
        "Research OAuth2 and summarize the findings.",
    ]

    for question in questions:
        # Generate a structured plan for each question.
        plan = create_llm_plan(question)

        print(f"\nQuestion: {question}")
        print("Agents:", plan.agents)
        print("Reason:", plan.reason)
