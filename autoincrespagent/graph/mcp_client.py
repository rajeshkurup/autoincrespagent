"""MCP client factory.

Returns a configured MultiServerMCPClient that spawns three MCP servers:
  - graphmcpserv      (graph DB tools)
  - mitigationmcpserv (mitigation workflow tools)
  - commsmcpserv      (communication channel tools)

As of langchain-mcp-adapters 0.2.x, MultiServerMCPClient is NOT a context
manager. Use the async get_tools() method directly:

    client = build_mcp_client()
    tools = await client.get_tools()
    graph = build_graph(tools)
    await graph.ainvoke(...)

All servers must be installed in the same venv:
    pip install -e ../graphmcpserv
    pip install -e ../mitigationmcpserv
    pip install -e ../commsmcpserv
"""

from langchain_mcp_adapters.client import MultiServerMCPClient

from autoincrespagent.config import settings


def build_mcp_client() -> MultiServerMCPClient:
    """Construct and return a MultiServerMCPClient for all MCP servers."""
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
        "mitigation": {
            "command": "python",
            "args": ["-m", "mcp_servers.mitigation.server"],
            "transport": "stdio",
            "cwd": settings.mitigationmcpserv_path,
            "env": {
                "QDRANT_URL": f"http://{settings.qdrant_host}:{settings.qdrant_port}",
                "OLLAMA_BASE_URL": settings.ollama_base_url,
                "CONFIDENCE_THRESHOLD": str(settings.confidence_threshold),
            },
        },
        "comms": {
            "command": "python",
            "args": ["-m", "mcp_servers.comms.server"],
            "transport": "stdio",
            "cwd": settings.commsmcpserv_path,
            "env": {
                "COMMS_LOG_DIR": "logs",
            },
        },
    })
