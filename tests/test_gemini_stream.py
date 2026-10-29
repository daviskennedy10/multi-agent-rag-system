
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load our Gemini API key and selected model.
load_dotenv()

# Initialize Google's Gemini client.
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Record when the request starts to measure latency.
start = time.perf_counter()

# Request a streaming response instead of a complete response.
stream = client.models.generate_content_stream(
    model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    contents="Explain dependency injection in three sentences.",
    config=types.GenerateContentConfig(
        # Limit output length to control costs.
        max_output_tokens=512,

        # Use the same thinking configuration as our earlier test.
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
    ),
)

# Track when the first visible text arrives.
first_token_received = False

# Process generated text chunks as Gemini produces them.
for chunk in stream:
    # Some chunks may contain metadata instead of text.
    token = chunk.text or ""

    if token:
        if not first_token_received:
            # Measure time to first visible text.
            elapsed = time.perf_counter() - start
            print(f"\nTime to first token: {elapsed:.2f}s\n")

            first_token_received = True

        # Print each chunk immediately.
        print(token, end="", flush=True)

# Measure the total generation time.
total = time.perf_counter() - start

print(f"\n\nTotal generation time: {total:.2f}s")
