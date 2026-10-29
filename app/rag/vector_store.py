
import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from app.rag.ingest import load_documents

from pathlib import Path

# Load API credentials from our .env file.
load_dotenv()

# Initialize our OpenAI API client.
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Locate the project root using this file's location.
# vector_store.py is inside app/rag/, so parents[2] is the root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Use CHROMA_PATH if configured, otherwise use the project's index.
# This works locally and inside our Docker container.
CHROMA_PATH = Path(
    os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_index"))
).resolve()

# Open our existing persistent Chroma database.
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

# Open the collection where documentation is stored.
collection = chroma_client.get_or_create_collection(
    name="developer_docs"
)


def embed_text(text: str) -> list[float]:
    # Convert a single query into an embedding.
    # We'll use this function during retrieval.
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return response.data[0].embedding


def reset_collection():
    # Delete the existing collection, including all old chunks.
    # WARNING: This permanently removes the collection's data.
    chroma_client.delete_collection(name="developer_docs")

    # Recreate an empty collection with the same name.
    global collection
    collection = chroma_client.get_or_create_collection(
        name="developer_docs"
    )

    # Confirm the new collection is empty.
    print(f"Database reset. Chunk count: {collection.count()}")

def index_documents(batch_size: int = 50, limit: int | None = None):
    # Load all documentation and split it into chunks.
    documents = load_documents("data/raw")

    # Optionally restrict how many chunks we index.
    # Useful for testing without processing the entire corpus.
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")

        documents = documents[:limit]

    # Calculate how many chunks need processing.
    total = len(documents)

    print(f"Found {total} chunks to index.")

    # Process documents in batches instead of individually.
    for start in range(0, total, batch_size):

        # Select the next batch of documents.
        batch = documents[start:start + batch_size]

        # Extract text from each chunk.
        texts = [doc["text"] for doc in batch]

        # Generate embeddings for the entire batch.
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )

        # Match returned embeddings to their input positions.
        embeddings = [None] * len(batch)

        for item in response.data:
            embeddings[item.index] = item.embedding

        # Store the chunks and their embeddings in Chroma.
        collection.upsert(
            ids=[doc["id"] for doc in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source": doc["source"],
                    "technology": doc["technology"],
                }
                for doc in batch
            ],
        )

        # Display progress after each completed batch.
        completed = min(start + batch_size, total)

        print(f"Indexed {completed}/{total} chunks.")

    print("Indexing completed successfully!")


if __name__ == "__main__":
    # Start with a clean database to avoid stale documents.
    reset_collection()

    # Index the complete documentation corpus in batches of 50.
    index_documents(batch_size=50)

    # Verify how many chunks were stored.
    print(f"Final database count: {collection.count()}")
