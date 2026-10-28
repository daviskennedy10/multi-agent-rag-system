
import os

# Import the AST tool that analyzes Python code structure.
from app.agents.code_tools import analyze_python_code

# Import the shared LangGraph state.
from app.agents.state import AgentState

# Reuse our configured OpenAI client.
from app.rag.vector_store import openai_client

# Import both code extraction and AST analysis tools.
from app.agents.code_tools import (
    analyze_python_code,
    extract_python_code,
)


def code_agent(state: AgentState) -> dict:
    # Read the user's question from LangGraph state.
    question = state["question"]

    try:
        # Extract the actual Python code from the user's message.
        code = extract_python_code(question)

        # Analyze only the extracted code, not the English instructions.
        analysis = (
            analyze_python_code(code)
            if code
            else {"valid": False, "error": "No Python code found"}
        )

        # Give the LLM a distinct code-analysis role.
        system_prompt = """
You are a specialized Python Code Analysis Agent.

Your responsibilities:
1. Explain Python code clearly and accurately.
2. Identify potential bugs and syntax errors.
3. Suggest improvements when appropriate.
4. Use the provided AST analysis as structural evidence.
5. Never claim code was executed or tested.
6. If no Python code is supplied, explain what is missing.
"""

        # Ask the LLM to interpret the code and AST findings.
        response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"User input:\n{question}\n\n"
                        f"AST analysis:\n{analysis}"
                    ),
                },
            ],
            temperature=0,
        )

        # Extract the generated code explanation.
        answer = response.choices[0].message.content or ""

        # Return the answer and execution trace to LangGraph.
        return {
            "answer": answer,
            "trace": state["trace"] + [
                "Code agent executed",
                "AST analysis completed",
            ],
        }

    except Exception as error:
        # Record failure without crashing the workflow.
        return {
            "answer": "Code analysis failed.",
            "trace": state["trace"] + [
                f"Code agent failed: {type(error).__name__}",
            ],
        }
