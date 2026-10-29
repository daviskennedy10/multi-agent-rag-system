
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load Tavily credentials from our local .env file.
load_dotenv()


def create_mcp_client() -> MultiServerMCPClient:
    # Read the API key without exposing it in logs.
    api_key = os.environ["TAVILY_API_KEY"]

    # Connect to Tavily's remote MCP server over HTTP.
    # Both Research and Code will use this same configuration.
    return MultiServerMCPClient(
        {
            "tavily": {
                "transport": "http",
                "url": (
                    "https://mcp.tavily.com/mcp/"
                    f"?tavilyApiKey={api_key}"
                ),
            }
        }
    )


async def get_tavily_tools():
    # Discover the tools exposed by Tavily's MCP server.
    client = create_mcp_client()

    # Return LangChain-compatible tool objects.
    return await client.get_tools()
