
from langgraph.graph import StateGraph, START, END

# Import our shared state definition.
from app.agents.state import AgentState

from app.agents.research import research_agent

from app.agents.code import code_agent

# Import our real Summary Agent implementation.
from app.agents.summary import summary_agent

from app.agents.planner import create_plan

# Import our structured LLM planner.
from app.agents.llm_planner import create_llm_plan

def supervisor(state: AgentState) -> dict:
    # Retrieve the existing execution plan.
    plan = state.get("plan", [])

    
    # FIRST VISIT: Generate an intelligent execution plan.
    if not plan:
        step = 0

        try:
            # Ask the LLM to select the appropriate agents.
            llm_plan = create_llm_plan(state["question"])

            # Extract the validated agent sequence.
            plan = llm_plan.agents

            # Record why the supervisor chose these agents.
            planning_event = (
                f"LLM supervisor plan: {plan}. "
                f"Reason: {llm_plan.reason}"
            )

        except Exception as error:
            # Fall back to our original keyword-based planner.
            # This keeps the workflow operational if OpenAI fails.
            plan = create_plan(state["question"])

            planning_event = (
                f"LLM planning failed: {type(error).__name__}. "
                f"Using fallback plan: {plan}"
            )

        # Record the planning decision in the execution trace.
        trace = state["trace"] + [planning_event]

    # SUBSEQUENT VISITS: Move to the next planned agent.
    else:
        step = state.get("step", 0) + 1
        trace = state["trace"]


    # COMPLETION: All planned agents have executed.
    if step >= len(plan):
        # Retrieve the outputs saved by individual agents.
        agent_results = state.get("agent_results", {})

        # Combine findings in their original execution order.
        sections = [
            f"## {agent.title()} Findings\n{agent_results[agent]}"
            for agent in plan
            if agent in agent_results
        ]

        # Combine multiple agent outputs into one answer.
        # Preserve the original answer for single-agent tasks.
        final_answer = (
            "\n\n".join(sections)
            if len(sections) > 1
            else state["answer"]
        )

        # Return the completed workflow state.
        # Do NOT access plan[step] here: the plan is finished.
        return {
            "plan": plan,
            "step": step,
            "answer": final_answer,
            "trace": trace + [
                "Supervisor completed plan"
            ],
        }

    # DELEGATION: More agents still need to execute.
    # Select the next worker from the execution plan.
    selected = plan[step]

    # Return the updated state so LangGraph can route correctly.
    return {
        "plan": plan,
        "step": step,
        "selected_agent": selected,
        "trace": trace + [
            f"Supervisor delegated to {selected}"
        ],
    }


def route_agent(state: AgentState) -> str:
    # Read the plan and step, using safe defaults.
    plan = state.get("plan", [])
    step = state.get("step", 0)
    # End the workflow when all planned agents have executed.
    if step >= len(plan):
        return "end"

    # Otherwise, execute the next selected worker.
    return state["selected_agent"]


# Create a graph whose shared state follows AgentState.
builder = StateGraph(AgentState)

# Register the supervisor and worker nodes.
builder.add_node("supervisor", supervisor)
builder.add_node("research", research_agent)
builder.add_node("code", code_agent)
builder.add_node("summary", summary_agent)

# Start execution at the supervisor.
builder.add_edge(START, "supervisor")


# Route to the next worker or end when the plan is complete.
builder.add_conditional_edges(
    "supervisor",
    route_agent,
    {
        "research": "research",
        "code": "code",
        "summary": "summary",
        "end": END,
    },
)

# Workers return to the supervisor for further coordination.
for agent in ["research", "code", "summary"]:
    builder.add_edge(agent, "supervisor")


# Compile the graph into an executable workflow.
graph = builder.compile()





if __name__ == "__main__":
    # This request requires both documentation research and code work.
    question = (
        "Research FastAPI dependency injection "
        "and implement a Python example."
    )

    # Start with an empty shared state.
    result = graph.invoke({
        "question": question,
        "selected_agent": "research",
        "answer": "",
        "trace": [],
        "agent_results": {},
    })

    # Inspect the execution plan.
    print("Plan:", result["plan"])

    # Verify that both agents saved their outputs.
    print("Completed agents:", list(result["agent_results"]))

    # Inspect the complete orchestration trace.
    print("\nTrace:")
    for event in result["trace"]:
        print("-", event)

    # Display a preview of the combined answer.
    print("\nFinal answer:")
    print(result["answer"][:800])


