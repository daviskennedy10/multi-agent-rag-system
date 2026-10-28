
import os

from dotenv import load_dotenv
from google import genai

# Load the Google API key from .env.
load_dotenv()

# Connect to Google's Gemini API.
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Display available models that support content generation.
for model in client.models.list():
    # Some models support only embeddings or other operations.
    actions = model.supported_actions or []

    if "generateContent" in actions:
        print(model.name)
