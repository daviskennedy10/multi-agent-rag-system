
import json
import os
import time

import requests

# Use the deployed Cloud Run endpoint.
BASE_URL = os.environ.get(
    "CLOUD_RUN_URL",
    "https://developer-docs-assistant-let7krnqfa-wl.a.run.app",
)

URL = f"{BASE_URL.rstrip('/')}/chat/stream"

question = "What is FastAPI dependency injection?"


def measure_request(label):
    # Record request start time.
    start = time.perf_counter()

    first_token = None
    completed = False

    # Stream the actual Cloud Run SSE response.
    with requests.post(
        URL,
        json={"question": question},
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()

        event_type = None

        # Process individual SSE lines as they arrive.
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                event_type = None
                continue

            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()

            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())

                if event_type == "token" and first_token is None:
                    first_token = time.perf_counter() - start

                if event_type == "done":
                    completed = True

    total = time.perf_counter() - start

    result = {
        "label": label,
        "ttft_seconds": (
            round(first_token, 3)
            if first_token is not None else None
        ),
        "total_seconds": round(total, 3),
        "completed": completed,
    }

    print(result, flush=True)
    return result


# Measure two consecutive requests.
results = [
    measure_request("first_request"),
    measure_request("warm_request"),
]

# Save measurements for the evaluation report.
with open("data/cloud_run_latency.json", "w") as file:
    json.dump(results, file, indent=2)
