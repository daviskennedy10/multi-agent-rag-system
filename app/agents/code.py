
import os

# Import the shared LangGraph state.
from app.agents.state import AgentState

from app.agents.llm_providers import generate_routed

# Import both code extraction and AST analysis tools.
from app.agents.code_tools import (
    analyze_python_code,
    extract_python_code,
)

# Allow our synchronous Code Agent to invoke async MCP tools.
import asyncio

# Reuse the exact same Tavily MCP search function.
from app.agents.web_tools import search_web

# Generate real model tokens through LangGraph custom events.
from app.agents.streaming import generate_streamed

def code_agent(state: AgentState) -> dict:
    # Read the user's question from LangGraph state.
    question = state["question"]

    # Identify requests that may need up-to-date API information.
    live_keywords = ["latest", "current", "recent", "2026"]

    needs_live_search = any(
        keyword in question.lower()
        for keyword in live_keywords
    )

    # Search for current documentation when appropriate.
    web_context = ""

    if needs_live_search:
        # Both Research and Code invoke the same MCP-backed tool.
        web_context = asyncio.run(search_web(question))


    # Retrieve findings from an earlier Research Agent execution.
    research_findings = state.get("agent_results", {}).get("research", "")

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

        
        # Build the Code Agent's prompt using the original request,
        # AST analysis, and any earlier Research findings.
        prompt = (
            f"{system_prompt}\n\n"
            f"User request:\n{question}\n\n"
            f"AST analysis:\n{analysis}\n\n"
            f"Research findings:\n{research_findings or 'None'}"
            f"\nLive web search results:\n{web_context or 'None'}"
        )

        # Generate the Code Agent's response incrementally.
        # LangGraph forwards each token to our SSE endpoint.
        result = generate_streamed(
            question=question,
            agent="code",
            prompt=prompt,
        )

        # Extract the generated answer and actual routing decision.
        answer = result["answer"]
        routing = result["routing"]


        # Return the analysis and inspectable provider information.
        return {
            "answer": answer,
            "routing": routing,
            "trace": state["trace"] + [
                # Record the Code Agent's execution.
                "Code agent executed",
                "AST analysis completed",

                # Record the MCP tool invocation when live search was used.
                *(
                    ["Code invoked MCP tool: tavily_search"]
                    if needs_live_search
                    else []
                ),

                # Record the actual LLM provider and model.
                f"LLM: {routing['provider']} / {routing['model']}",

                # Explain why the router selected that provider.
                f"Routing reason: {routing['reason']}",
            ],
            "agent_results": {
                **state.get("agent_results", {}),
                "code": answer,
            },
        }

    except Exception as error:
        # Record failure without crashing the workflow.
        return {
            "answer": "Code analysis failed.",
            "trace": state["trace"] + [
                f"Code agent failed: {type(error).__name__}",
            ],
        }
