
# Import our shared MCP tool-discovery function.
from app.agents.mcp_tools import get_tavily_tools


async def search_web(query: str) -> str:
    # Connect to Tavily and discover available MCP tools.
    tools = await get_tavily_tools()

    # Find the Tavily search tool by its exact name.
    search_tool = next(
        tool for tool in tools
        if tool.name == "tavily_search"
    )

    # Invoke the remote MCP search tool.
    # The query is passed as a structured tool argument.
    result = await search_tool.ainvoke({
        "query": query,
    })

    # Convert the tool response into text for our agents.
    return str(result)
