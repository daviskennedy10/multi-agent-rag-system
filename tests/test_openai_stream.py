
import os

from dotenv import load_dotenv
from openai import OpenAI

# Load our OpenAI API key and model configuration.
load_dotenv()

# Initialize the OpenAI client.
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Request a streaming response from the model.
stream = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    messages=[
        {
            "role": "user",
            "content": "Explain dependency injection in three sentences.",
        }
    ],
    stream=True,
)

# Process text as the model generates it.
for chunk in stream:
    # Some chunks contain metadata rather than text.
    token = chunk.choices[0].delta.content

    if token:
        # Print immediately without waiting for the full answer.
        print(token, end="", flush=True)

print()
