
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
)


async def evaluate_additional_metrics():
    # Load our existing API credentials.
    load_dotenv()

    # Read previously generated RAG answers and contexts.
    results_path = Path("data/evaluation_results.json")
    results = json.loads(results_path.read_text(encoding="utf-8"))

    # Configure the evaluation LLM and embedding model.
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    evaluator_llm = llm_factory(
        "gpt-4.1-mini",
        client=client,
        max_tokens=4096,
    )

    embeddings = OpenAIEmbeddings(
        client=client,
        model="text-embedding-3-small",
    )

    # Initialize the two additional evaluation metrics.
    relevancy_metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=embeddings,
    )

    precision_metric = ContextPrecisionWithoutReference(
        llm=evaluator_llm,
    )

    # Evaluate each previously generated answer.
    for result in results:
        question_id = result["id"]

        # Skip entries without successful RAG generation.
        if not result.get("answer") or not result.get("contexts"):
            print(f"Skipping {question_id}: missing answer or contexts.")
            continue

        print(f"\nEvaluating {question_id}", flush=True)

        # Evaluate answer relevancy only when missing.
        if result.get("answer_relevancy") is None:
            try:
                score = await relevancy_metric.ascore(
                    user_input=result["question"],
                    response=result["answer"],
                )

                result["answer_relevancy"] = score.value

                print(f"Answer relevancy: {score.value:.4f}")

            except Exception as error:
                # Record metric failure without losing other results.
                result["answer_relevancy_error"] = str(error)
                print(f"Answer relevancy failed: {error}")

        # Evaluate context precision only when missing.
        if result.get("context_precision") is None:
            try:
                score = await precision_metric.ascore(
                    user_input=result["question"],
                    response=result["answer"],
                    retrieved_contexts=result["contexts"],
                )

                result["context_precision"] = score.value

                print(f"Context precision: {score.value:.4f}")

            except Exception as error:
                # Preserve other metric results if this one fails.
                result["context_precision_error"] = str(error)
                print(f"Context precision failed: {error}")

        # Save after each question to preserve partial progress.
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Calculate averages for each available metric.
    for metric_name in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    ]:
        scores = [
            item[metric_name]
            for item in results
            if item.get(metric_name) is not None
        ]

        if scores:
            average = sum(scores) / len(scores)
            print(
                f"\nAverage {metric_name}: {average:.4f} "
                f"({len(scores)}/{len(results)} questions)"
            )

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    # Run the evaluation asynchronously.
    asyncio.run(evaluate_additional_metrics())
