from pathlib import Path


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks



def load_documents(directory: str) -> list[dict]:
    # Root directory containing all documentation.
    root = Path(directory)

    # Store the processed document chunks.
    documents = []

    # Recursively find all .txt files.
    for path in sorted(root.rglob("*.txt")):

        # Read the document's contents.
        text = path.read_text(encoding="utf-8")

        # Ignore empty documents.
        if not text.strip():
            continue

        # Get the relative path, e.g. fastapi/authentication.txt.
        relative_path = path.relative_to(root)

        # Extract the technology name from the first folder.
        technology = (
            relative_path.parts[0]
            if len(relative_path.parts) > 1
            else "general"
        )

        # Split the document into overlapping chunks.
        for index, chunk in enumerate(chunk_text(text)):

            # Create an ID containing the full relative path.
            # This prevents collisions between different folders.
            chunk_id = f"{relative_path.as_posix()}::{index}"

            # Store the chunk and its source information.
            documents.append({
                "id": chunk_id,
                "source": str(relative_path),
                "technology": technology,
                "text": chunk,
            })

    # Return all chunks ready for embedding.
    return documents



if __name__ == "__main__":
    documents = load_documents("data/raw")

    print(f"Total chunks: {len(documents)}")

    for document in documents:
        print(f"\nID: {document['id']}")
        print(f"Source: {document['source']}")
        print(f"Characters: {len(document['text'])}")
        print(f"Preview: {document['text'][:100]}")