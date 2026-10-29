
import json
import time
from pathlib import Path

# Import our existing RAG pipeline.
from app.rag.generate import generate_with_context


# Load the fixed evaluation questions.
questions = json.loads(
    Path("data/evaluation_questions.json").read_text(
        encoding="utf-8"
    )
)

results = []

for item in questions:
    # Start measuring immediately before RAG execution.
    start = time.perf_counter()

    try:
        # Run retrieval and generation without LangGraph streaming.
        result = generate_with_context(item["question"])

        # Measure elapsed time in seconds.
        latency = time.perf_counter() - start

        results.append({
            "id": item["id"],
            "latency_seconds": round(latency, 3),
            "provider": (
                result["routing"]["provider"]
                if result.get("routing")
                else None
            ),
            "success": True,
        })

        print(
            f"{item['id']}: {latency:.2f}s "
            f"({results[-1]['provider']})",
            flush=True,
        )

    except Exception as error:
        # Record failures instead of hiding them.
        results.append({
            "id": item["id"],
            "success": False,
            "error": type(error).__name__,
        })

        print(f"{item['id']}: FAILED", flush=True)

# Save the raw measurements for our final report.
output = Path("data/latency_results.json")
output.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

# Calculate the mean over successful requests only.
latencies = [
    row["latency_seconds"]
    for row in results
    if row["success"]
]

if latencies:
    average = sum(latencies) / len(latencies)
    print(f"\nAverage RAG latency: {average:.2f}s")
    print(f"Successful requests: {len(latencies)}/{len(questions)}")
else:
    print("\nNo successful latency measurements.")
