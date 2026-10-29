
import asyncio
import traceback

from app.agents.mcp_tools import get_tavily_tools


async def main():
    # Confirm that our test actually starts.
    print("Starting Tavily MCP connection test...", flush=True)

    try:
        # Limit tool discovery to 20 seconds.
        # This prevents the test from hanging indefinitely.
        tools = await asyncio.wait_for(
            get_tavily_tools(),
            timeout=20,
        )

        # Display the discovered MCP tools.
        print(f"Discovered {len(tools)} MCP tools:", flush=True)

        for tool in tools:
            print("-", tool.name)

    except asyncio.TimeoutError:
        # Report when the MCP handshake takes too long.
        print("ERROR: MCP connection timed out after 20 seconds.")

    except Exception:
        # Display the full error to identify connection failures.
        traceback.print_exc()


if __name__ == "__main__":
    # Start the asynchronous connection test.
    asyncio.run(main())
