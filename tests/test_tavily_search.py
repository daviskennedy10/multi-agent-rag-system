
import asyncio

# Import our shared web-search function.
from app.agents.web_tools import search_web


async def main():
    # Search for current FastAPI documentation information.
    query = "FastAPI latest release official documentation"

    print("Searching Tavily...", flush=True)

    # Execute a real search through Tavily Remote MCP.
    result = await search_web(query)

    # Display a preview to avoid overwhelming terminal output.
    print("\nSEARCH RESULT:")
    print(result[:1500])


if __name__ == "__main__":
    # Run the asynchronous search test.
    asyncio.run(main())
