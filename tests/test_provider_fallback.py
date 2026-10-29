
from unittest.mock import patch

from google.genai.errors import ClientError

# Import the module so we can replace its API clients temporarily.
from app.agents import llm_providers


def fake_gemini_stream(*args, **kwargs):
    # Simulate Gemini rejecting the request due to exhausted quota.
    raise ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Simulated quota exceeded",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )

    # Make this a generator, matching Gemini's streaming interface.
    yield


def fake_openai_stream(*args, **kwargs):
    # Simulate OpenAI successfully streaming a response.
    from types import SimpleNamespace

    yield SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="Fallback succeeded!")
            )
        ]
    )


# Replace both real provider calls with local mock functions.
# No API requests are made during this test.
with (
    patch.object(
        llm_providers.gemini_client.models,
        "generate_content_stream",
        side_effect=fake_gemini_stream,
    ),
    patch.object(
        llm_providers.openai_client.chat.completions,
        "create",
        side_effect=fake_openai_stream,
    ),
):
    # Run our unified streaming interface.
    events = list(
        llm_providers.stream_routed(
            question="What is dependency injection?",
            agent="research",
            prompt="Explain dependency injection.",
        )
    )

    # Display every event for inspection.
    for event in events:
        print(event)

    # Verify the final provider is OpenAI.
    assert events[-1]["type"] == "complete"
    assert events[-1]["data"]["provider"] == "openai"

    # Verify the simulated answer was streamed.
    assert any(
        event["type"] == "token"
        and event["data"] == "Fallback succeeded!"
        for event in events
    )

print("\nProvider fallback test passed!")
