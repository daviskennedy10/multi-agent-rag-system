
from pathlib import Path

# LangChain utility for splitting text at natural boundaries.
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[str]:
    # Validate the chunking configuration.
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    # Prefer splitting at paragraphs, lines, and spaces.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    # Return the generated text chunks.
    return splitter.split_text(text)


def load_documents(directory: str) -> list[dict]:
    # Root folder containing our documentation.
    root = Path(directory)

    # Store all processed chunks here.
    documents = []

    # Recursively find Markdown and plain-text files.
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )

    for path in paths:
        # Read each document as UTF-8 text.
        text = path.read_text(encoding="utf-8")

        # Ignore empty files.
        if not text.strip():
            continue

        # Example: fastapi/tutorial/dependencies.md
        relative_path = path.relative_to(root)

        # The first folder identifies the technology.
        technology = (
            relative_path.parts[0]
            if len(relative_path.parts) > 1
            else "general"
        )

        # Split the document into smaller, overlapping chunks.
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            # Use the full relative path to avoid ID collisions.
            chunk_id = f"{relative_path.as_posix()}::{index}"

            # Preserve the text and its source information.
            documents.append({
                "id": chunk_id,
                "source": str(relative_path),
                "technology": technology,
                "text": chunk,
            })

    return documents


if __name__ == "__main__":
    # Load and chunk all available documentation.
    documents = load_documents("data/raw")

    # Show how much documentation we processed.
    print(f"Total chunks: {len(documents)}")

    # Display only the first three chunks to avoid huge output.
    for document in documents[:3]:
        print(f"\nID: {document['id']}")
        print(f"Technology: {document['technology']}")
        print(f"Characters: {len(document['text'])}")
        print(f"Preview: {document['text'][:150]}")
