
# Import our shared provider-streaming function.
from app.agents.llm_providers import stream_routed


# Test a simple question that should select Gemini.
question = "What is dependency injection?"

# Request streaming generation through our router.
events = stream_routed(
    question=question,
    agent="research",
    prompt="Explain dependency injection in two sentences.",
)

# Display each event as it arrives.
for event in events:
    if event["type"] == "routing":
        # Show the selected provider.
        print("\nROUTING:", event["data"])

    elif event["type"] == "token":
        # Display text immediately.
        print(event["data"], end="", flush=True)

    elif event["type"] == "complete":
        # Confirm generation finished.
        print("\n\nStreaming completed.")
