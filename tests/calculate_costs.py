
import json
from pathlib import Path

# Read the token counts measured during our benchmark.
path = Path("data/detailed_benchmark.json")
results = json.loads(path.read_text(encoding="utf-8"))

# Example rates in USD per million tokens.
# Replace with verified rates for your exact model and pricing tier.
INPUT_RATE = 0.40
OUTPUT_RATE = 1.60

total_cost = 0.0

for row in results:
    # Failed requests or missing token usage cannot be priced here.
    if not row["success"]:
        continue

    if row["provider"] != "openai":
        continue

    input_tokens = row.get("input_tokens")
    output_tokens = row.get("output_tokens")

    if input_tokens is None or output_tokens is None:
        print(f"{row['id']}: missing token usage")
        continue

    # Convert per-million-token rates into per-request cost.
    cost = (
        input_tokens * INPUT_RATE
        + output_tokens * OUTPUT_RATE
    ) / 1_000_000

    row["estimated_cost_usd"] = round(cost, 8)
    total_cost += cost

    print(
        f"{row['id']}: "
        f"{input_tokens} input tokens, "
        f"{output_tokens} output tokens, "
        f"${cost:.6f}"
    )

# Preserve cost calculations alongside the benchmark measurements.
path.write_text(json.dumps(results, indent=2))

print(f"\nTotal estimated OpenAI cost: ${total_cost:.6f}")
