
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall


async def main():
    # Load credentials and existing evaluation artifacts.
    load_dotenv()

    questions = json.loads(
        Path("data/evaluation_questions.json").read_text()
    )

    results_path = Path("data/evaluation_results.json")
    results = json.loads(results_path.read_text())

    # Match each question with its verified reference answer.
    references = {
        item["id"]: item["reference"]
        for item in questions
    }

    # Configure the RAGAS evaluation model.
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    llm = llm_factory(
        "gpt-4.1-mini",
        client=client,
        max_tokens=4096,
    )

    metric = ContextRecall(llm=llm)

    # Reuse saved retrieval contexts instead of generating answers again.
    for result in results:
        question_id = result["id"]

        if result.get("context_recall") is not None:
            print(f"Skipping {question_id}: already evaluated.")
            continue

        if not result.get("contexts"):
            print(f"Skipping {question_id}: no contexts.")
            continue

        try:
            # Measure how much reference-answer information
            # was captured by the retrieved documentation.
            score = await metric.ascore(
                user_input=result["question"],
                reference=references[question_id],
                retrieved_contexts=result["contexts"],
            )

            result["context_recall"] = score.value
            print(f"{question_id}: {score.value:.4f}", flush=True)

        except Exception as error:
            # Preserve failures for later investigation.
            result["context_recall_error"] = str(error)
            print(f"{question_id}: FAILED - {error}", flush=True)

        # Save after every question to preserve progress.
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False)
        )

    # Report average recall across successful evaluations.
    scores = [
        row["context_recall"]
        for row in results
        if row.get("context_recall") is not None
    ]

    if scores:
        print(
            f"\nAverage context recall: "
            f"{sum(scores) / len(scores):.4f}"
        )
        print(f"Successful evaluations: {len(scores)}/{len(questions)}")


if __name__ == "__main__":
    asyncio.run(main())
