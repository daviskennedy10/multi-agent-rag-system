
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# RAGAS components for faithfulness evaluation.
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# Our existing RAG pipeline.
from app.rag.generate import generate_with_context


async def evaluate_dataset():
    # Load environment variables, including our OpenAI API key.
    load_dotenv()

    # Read the fixed evaluation questions from JSON.
    dataset_path = Path("data/evaluation_questions.json")
    questions = json.loads(dataset_path.read_text(encoding="utf-8"))

    # Load previous results so we don't repeat successful evaluations.
    output_path = Path("data/evaluation_results.json")

    previous_results = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else []
    )

    # Match previous results to their question IDs.
    previous_by_id = {
        result["id"]: result
        for result in previous_results
    }

    # Keep successful evaluations and retry only failed ones.
    results = [
        result
        for result in previous_results
        if result.get("faithfulness") is not None
    ]

    # Configure the LLM that will judge answer faithfulness.
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    evaluator_llm = llm_factory(
        "gpt-4.1-mini",
        client=client,
        max_tokens=4096,
    )

    # Inspect the evaluator's type and configuration.
    # This does not make an API request.
    print("Evaluator type:", type(evaluator_llm))
    print("Evaluator configuration:", evaluator_llm)

    # Create our RAGAS faithfulness evaluator.
    metric = Faithfulness(llm=evaluator_llm)

    # Evaluate each question using the same RAG pipeline.
    for index, item in enumerate(questions, start=1):
        question = item["question"]

        # Skip questions that already have successful evaluation results.
        previous = previous_by_id.get(item["id"])

        if previous and previous.get("faithfulness") is not None:
            print(f"Skipping {item['id']}: already evaluated.")
            continue

        print(f"\nEvaluating {index}/{len(questions)}")
        print(f"Question: {question}", flush=True)

        try:
            # Generate an answer and preserve retrieved evidence.
            # to_thread prevents the synchronous generator from
            # blocking our asynchronous evaluation function.
            rag_result = await asyncio.to_thread(
                generate_with_context,
                question,
            )

            # Judge whether generated claims are supported by context.
            score = await metric.ascore(
                user_input=question,
                response=rag_result["answer"],
                retrieved_contexts=rag_result["contexts"],
            )

            # Save the score, answer, and retrieved source IDs.
            results.append({
                "id": item["id"],
                "question": question,
                "answer": rag_result["answer"],
                "faithfulness": score.value,
                "contexts": rag_result["contexts"],
                "sources": [
                    chunk["id"] for chunk in rag_result["chunks"]
                ],
                "citation_validation": rag_result["citation_validation"],
            })

            print(f"Faithfulness: {score.value:.4f}", flush=True)

        except Exception as error:
            # Record failures instead of silently skipping questions.
            results.append({
                "id": item["id"],
                "question": question,
                "faithfulness": None,
                "error": str(error),
            })

            print(f"Evaluation failed: {error}", flush=True)

    # Create the output directory if it doesn't exist.
    output_path = Path("data/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the full results for future inspection.
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Calculate the average across successful evaluations.
    scores = [
        result["faithfulness"]
        for result in results
        if result["faithfulness"] is not None
    ]

    if scores:
        average = sum(scores) / len(scores)

        print(f"\nAverage faithfulness: {average:.4f}")
        print(f"Successful evaluations: {len(scores)}/{len(questions)}")
    else:
        print("\nNo evaluations succeeded.")

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Execute the asynchronous evaluation workflow.
    asyncio.run(evaluate_dataset())
