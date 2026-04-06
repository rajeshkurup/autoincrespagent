"""MCP client factory.

Returns a configured MultiServerMCPClient and an async helper to load
all tools from the graphmcpserv subprocess.

As of langchain-mcp-adapters 0.2.x, MultiServerMCPClient is NOT a context
manager. Use the async get_tools() method directly:

    client = build_mcp_client()
    tools = await client.get_tools()
    graph = build_graph(tools)
    await graph.ainvoke(...)

graphmcpserv must be installed in the same venv:
    pip install -e ../graphmcpserv
"""

from langchain_mcp_adapters.client import MultiServerMCPClient

from autoincrespagent.config import settings


def build_mcp_client() -> MultiServerMCPClient:
    """Construct and return a MultiServerMCPClient for graphmcpserv."""
    return MultiServerMCPClient({
        "graph_db": {
            "command": "python",
            "args": ["-m", "mcp_servers.graph_db.server"],
            "transport": "stdio",
            "env": {
                "GRAPHSERV_URL": settings.graphserv_url,
                "GRAPHSERV_TIMEOUT": "10.0",
            },
        },
    })
