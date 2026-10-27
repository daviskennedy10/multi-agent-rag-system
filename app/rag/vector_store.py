import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from app.rag.ingest import load_documents

load_dotenv()

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

chroma_client = chromadb.PersistentClient(path="./chroma_index")

collection = chroma_client.get_or_create_collection(
    name="developer_docs"
)


def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def index_documents():
    documents = load_documents("data/raw")

    for document in documents:
        embedding = embed_text(document["text"])

        collection.upsert(
            ids=[document["id"]],
            embeddings=[embedding],
            documents=[document["text"]],
            metadatas=[{"source": document["source"]}],
        )

    print(f"Indexed {len(documents)} chunks successfully!")


if __name__ == "__main__":
    index_documents()