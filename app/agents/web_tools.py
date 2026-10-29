
import json

# Import our shared Tavily MCP connection.
from app.agents.mcp_tools import get_tavily_tools


async def search_web(query: str) -> str:
    # Discover tools exposed by Tavily's MCP server.
    tools = await get_tavily_tools()

    # Select the same MCP tool for both agents.
    search_tool = next(
        tool for tool in tools
        if tool.name == "tavily_search"
    )

    # Execute a real search through Tavily Remote MCP.
    result = await search_tool.ainvoke({"query": query})

    # Convert the result into text for the LLM.
    if isinstance(result, str):
        return result

    # Preserve structured search results as JSON text.
    return json.dumps(result, ensure_ascii=False, default=str)
