
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from google import genai

from app.rag.retriever import retrieve

load_dotenv()

questions = json.loads(
    Path("data/evaluation_questions.json").read_text()
)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

results = []

for item in questions:
    question = item["question"]

    # Measure retrieval independently from model generation.
    retrieval_start = time.perf_counter()
    chunks = retrieve(question, k=5)
    retrieval_seconds = time.perf_counter() - retrieval_start

    # Build the same evidence context for both providers.
    context = "\n\n".join(
        f"[{chunk['id']}]\n{chunk['text']}"
        for chunk in chunks
    )

    prompt = (
        "Answer using only the documentation below. "
        "Cite the supplied chunk IDs.\n\n"
        f"{context}\n\nQuestion: {question}"
    )

    for provider in ["openai", "google"]:
        start = time.perf_counter()
        first_token = None
        input_tokens = None
        output_tokens = None
        success = False
        error_name = None

        try:
            if provider == "openai":
                # Stream OpenAI output and collect usage at completion.
                stream = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    stream_options={"include_usage": True},
                )

                for chunk in stream:
                    if chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens

                    for choice in chunk.choices:
                        token = choice.delta.content

                        if token and first_token is None:
                            first_token = time.perf_counter() - start

                success = True

            else:
                # Stream Gemini output and collect usage metadata.
                stream = gemini_client.models.generate_content_stream(
                    model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
                    contents=prompt,
                )

                for chunk in stream:
                    if chunk.text and first_token is None:
                        first_token = time.perf_counter() - start

                    if chunk.usage_metadata:
                        input_tokens = (
                            chunk.usage_metadata.prompt_token_count
                        )
                        output_tokens = (
                            chunk.usage_metadata.candidates_token_count
                        )

                success = True

        except Exception as error:
            # Keep failures visible rather than silently excluding them.
            error_name = type(error).__name__

        llm_seconds = time.perf_counter() - start

        row = {
            "id": item["id"],
            "provider": provider,
            "success": success,
            "retrieval_seconds": round(retrieval_seconds, 3),
            "llm_seconds": round(llm_seconds, 3),
            "total_seconds": round(
                retrieval_seconds + llm_seconds, 3
            ),
            "ttft_seconds": (
                round(first_token, 3)
                if first_token is not None else None
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": error_name,
        }

        results.append(row)

        print(
            f"{item['id']} | {provider} | "
            f"success={success} | "
            f"total={row['total_seconds']}s | "
            f"TTFT={row['ttft_seconds']}s",
            flush=True,
        )

        # Preserve partial results after each provider test.
        output = Path("data/detailed_benchmark.json")
        output.write_text(json.dumps(results, indent=2))

print("\nDetailed benchmark completed.")
