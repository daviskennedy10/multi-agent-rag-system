
import os

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

import time

from app.agents.llm_router import route_llm

# Load API keys and model settings from .env.
load_dotenv()

# Initialize clients for our two independent LLM providers.
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])




def generate_with_openai(prompt: str) -> str:
    # Send the prompt to OpenAI.
    response = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    # Extract the generated text.
    return response.choices[0].message.content or ""


def generate_with_gemini(prompt: str) -> str:
    # Record when the API request begins.
    start = time.perf_counter()

    print("Sending request to Gemini...", flush=True)

    try:
        # Request a short response to test basic API connectivity.
        response = gemini_client.models.generate_content(
            model=os.environ["GEMINI_MODEL"],
            contents=prompt,
            config={
                "max_output_tokens": 1024,
                "thinking_config": {
                    "thinking_budget": 0,
                },
            },
        )

        # Calculate the total API response time.
        elapsed = time.perf_counter() - start
        print(f"Gemini response time: {elapsed:.2f}s")

        # Return the generated text.
        return response.text or ""

    except Exception as error:
        # Display the error so we can diagnose API failures.
        print(f"Gemini error: {type(error).__name__}: {error}")
        raise

def generate_routed(
    question: str,
    agent: str,
    prompt: str,
) -> dict:
    # Select the provider based on task complexity.
    decision = route_llm(question, agent)

    # Generate using the selected provider.
    if decision["provider"] == "google":
        try:
            answer = generate_with_gemini(prompt)

        except Exception as error:
            # Fall back to OpenAI if Gemini fails.
            answer = generate_with_openai(prompt)

            # Record the provider that actually generated the answer.
            decision = {
                "provider": "openai",
                "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                "reason": (
                    f"Fallback after Gemini failure: "
                    f"{type(error).__name__}"
                ),
            }

    else:
        # OpenAI handles requests routed to it directly.
        answer = generate_with_openai(prompt)

    # Return both the answer and the routing decision.
    return {
        "answer": answer,
        "routing": decision,
    }


if __name__ == "__main__":
    # Use the same question to verify both providers.
    question = "Explain dependency injection in one sentence."

    # Test OpenAI generation.
    print("\nOPENAI:")
    print(generate_with_openai(question))

    # Test Gemini generation.
    print("\nGEMINI:")
    print(generate_with_gemini(question))
