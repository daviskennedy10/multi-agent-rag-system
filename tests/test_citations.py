
from app.rag.citations import validate_citations


# Simulate a chunk retrieved from our documentation.
chunks = [
    {
        "id": "fastapi/tutorial/security/first-steps.md::0",
        "text": "FastAPI supports OAuth2 authentication.",
    }
]

# This claim is NOT supported by the retrieved chunk.
# However, the citation ID is valid.
answer = (
    "FastAPI automatically encrypts every database "
    "password using AES-256. "
    "[fastapi/tutorial/security/first-steps.md::0]"
)

# Check whether our validator detects the unsupported claim.
result = validate_citations(answer, chunks)

print(result)
