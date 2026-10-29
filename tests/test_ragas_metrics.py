
import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Import modern RAGAS interfaces.
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
)

# Load our OpenAI credentials.
load_dotenv()

# Create an asynchronous OpenAI client.
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Configure the LLM used to judge evaluation results.
evaluator_llm = llm_factory(
    "gpt-4.1-mini",
    client=client,
    max_tokens=4096,
)

# Configure embeddings for semantic answer relevancy.
embeddings = OpenAIEmbeddings(
    client=client,
    model="text-embedding-3-small",
)


async def main():
    # Use a small fixed example to verify metric compatibility.
    question = "What is FastAPI dependency injection?"

    answer = (
        "FastAPI dependency injection uses Depends() "
        "to automatically provide reusable dependencies "
        "to path operation functions."
    )

    contexts = [
        (
            "FastAPI provides a dependency injection system "
            "that resolves dependencies declared using Depends()."
        ),
        (
            "FastAPI automatically generates OpenAPI documentation."
        ),
    ]

    # Initialize the answer relevancy evaluator.
    relevancy_metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=embeddings,
    )

    # Initialize reference-free context precision.
    precision_metric = ContextPrecisionWithoutReference(
        llm=evaluator_llm,
    )

    # Evaluate whether the answer addresses the question.
    relevancy = await relevancy_metric.ascore(
        user_input=question,
        response=answer,
    )

    # Evaluate whether useful contexts rank highly.
    precision = await precision_metric.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    # Display both scores.
    print(f"Answer relevancy: {relevancy.value:.4f}")
    print(f"Context precision: {precision.value:.4f}")


if __name__ == "__main__":
    # Execute both asynchronous evaluations.
    asyncio.run(main())
