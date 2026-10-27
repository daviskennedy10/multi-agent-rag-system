
import re


def validate_citations(answer: str, chunks: list[dict]) -> dict:
    # Collect the exact IDs of chunks retrieved from Chroma.
    valid_ids = {chunk["id"] for chunk in chunks}

    # Find bracketed citations containing our chunk-ID separator.
    # Example: [fastapi/tutorial/request-files.md::0]
    citations = re.findall(r"\[([^\[\]\n]*::[^\[\]\n]*)\]", answer)

    # Identify citations that do not match a retrieved chunk ID.
    invalid = [
        citation
        for citation in citations
        if citation not in valid_ids
    ]

    # Return the validation results.
    return {
        "valid": len(invalid) == 0,
        "citations": citations,
        "invalid": invalid,
    }


if __name__ == "__main__":
    # Simulate a retrieved chunk.
    chunks = [{"id": "fastapi/tutorial/request-files.md::0"}]

    # This citation refers to a real retrieved chunk.
    answer = "FastAPI supports file uploads [fastapi/tutorial/request-files.md::0]."

    # Check whether the citation is valid.
    print(validate_citations(answer, chunks))
