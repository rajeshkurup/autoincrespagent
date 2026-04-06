"""CLI entry point for the incident response agent.

Usage:
    # Poll graphserv every POLL_INTERVAL_SECONDS for active anomalies:
    python trigger.py --poll

    # Run once (useful for ad-hoc testing without MySQL):
    python trigger.py

Prerequisites:
    1. Ollama running:    ollama serve
    2. graphserv running: cd ../graphserv && make run
    3. Neo4j running
    4. (Optional) MySQL running for persistent checkpointing
    5. graphmcpserv installed: pip install -e ../graphmcpserv
"""

import argparse
import asyncio
import json
import logging
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from autoincrespagent.config import settings
from autoincrespagent.graph.mcp_client import build_mcp_client
from autoincrespagent.graph.workflow import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _initial_state(session_id: str) -> dict:
    return {
        "phase": "detect",
        "session_id": session_id,
        "feedback_iteration": 0,
        "incident_id": None,
        "severity": None,
        "anomaly_nodes": [],
        "root_causes": [],
        "mitigation_workflows": [],
        "mitigation_confidence": 0.0,
        "communications_sent": [],
        "incident_summary": None,
        "messages": [],
        "feedback_request": None,
    }


async def run_once(graph) -> dict:
    session_id = str(uuid.uuid4())
    logger.info(f"trigger: starting session {session_id}")
    config = {"configurable": {"thread_id": session_id}}
    result = await graph.ainvoke(_initial_state(session_id), config=config)
    logger.info(f"trigger: session {session_id} complete — phase={result.get('phase')}")
    return result


async def poll_loop(graph) -> None:
    interval = settings.poll_interval_seconds
    logger.info(f"trigger: polling every {interval}s — press Ctrl+C to stop")
    while True:
        try:
            await run_once(graph)
        except Exception as exc:
            logger.error(f"trigger: poll iteration failed: {exc}")
        await asyncio.sleep(interval)


async def _build_graph_with_mysql(all_tools: list):
    """Try to wire in MySQL checkpointer; fall back to in-memory."""
    try:
        import aiomysql
        from autoincrespagent.memory.mysql_saver import MySQLSaver

        pool = await aiomysql.create_pool(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            db=settings.mysql_database,
            autocommit=False,
        )
        logger.info("trigger: using MySQL checkpoint saver")
        return build_graph(all_tools, checkpointer=MySQLSaver(pool)), pool
    except Exception as exc:
        logger.warning(f"trigger: MySQL unavailable ({exc}) — using in-memory checkpointer")
        return build_graph(all_tools), None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Incident response agent")
    parser.add_argument("--poll", action="store_true", help="Poll continuously")
    args = parser.parse_args()

    client = build_mcp_client()
    all_tools = await client.get_tools()
    logger.info(f"trigger: loaded {len(all_tools)} MCP tools: {[t.name for t in all_tools]}")

    graph, pool = await _build_graph_with_mysql(all_tools)

    try:
        if args.poll:
            await poll_loop(graph)
        else:
            result = await run_once(graph)
            # Print summary (exclude raw messages from output)
            summary = {k: v for k, v in result.items() if k != "messages"}
            print(json.dumps(summary, indent=2, default=str))
    finally:
        if pool:
            pool.close()
            await pool.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
