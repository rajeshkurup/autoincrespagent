"""StateGraph assembly.

Wires agent nodes and conditional edges together into a compiled graph.
Only incident_detector is fully implemented in this phase; the remaining
nodes are stubs that will be replaced as each phase is built out.
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from autoincrespagent.agents.incident_detector import make_incident_detector
from autoincrespagent.agents.state import AgentState
from autoincrespagent.agents.supervisor import supervisor

logger = logging.getLogger(__name__)

# Tool names needed by each agent (for filtering the full tool list)
_GRAPH_DB_TOOL_NAMES = {
    "list_anomalies",
    "get_node",
    "root_cause_analysis",
    "blast_radius",
    "get_relationships",
    "create_incident_ticket",
    "link_incident_to_node",
    "get_rca_tickets",
    "get_change_tickets",
    "update_node_status",
}


# ── Placeholder nodes for future phases ──────────────────────────────

async def _root_cause_finder_stub(state: AgentState) -> dict:
    """Placeholder — Phase 2 will replace this with the real agent."""
    logger.info("root_cause_finder: not yet implemented — ending workflow")
    return {"phase": "done"}


async def _incident_mitigator_stub(state: AgentState) -> dict:
    logger.info("incident_mitigator: not yet implemented — ending workflow")
    return {"phase": "done"}


async def _incident_communicator_stub(state: AgentState) -> dict:
    logger.info("incident_communicator: not yet implemented — ending workflow")
    return {"phase": "done"}


async def _incident_summarizer_stub(state: AgentState) -> dict:
    logger.info("incident_summarizer: not yet implemented — ending workflow")
    return {"phase": "done"}


# ── Graph builder ─────────────────────────────────────────────────────

def build_graph(all_tools: list, checkpointer=None):
    """Compile and return the incident response StateGraph.

    Args:
        all_tools:    List of LangChain BaseTool instances from the MCP client.
        checkpointer: LangGraph checkpointer (defaults to MemorySaver for dev).

    Returns:
        A compiled LangGraph graph ready for ainvoke / astream.
    """
    graph_tools = [t for t in all_tools if t.name in _GRAPH_DB_TOOL_NAMES]
    logger.info(f"build_graph: loaded {len(graph_tools)} graph DB tools")

    builder = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────
    builder.add_node("incident_detector",      make_incident_detector(graph_tools))
    builder.add_node("root_cause_finder",      _root_cause_finder_stub)
    builder.add_node("incident_mitigator",     _incident_mitigator_stub)
    builder.add_node("incident_communicator",  _incident_communicator_stub)
    builder.add_node("incident_summarizer",    _incident_summarizer_stub)

    # ── Entry point ───────────────────────────────────────────────────
    builder.set_entry_point("incident_detector")

    # ── Conditional edges (all route through supervisor) ──────────────
    builder.add_conditional_edges("incident_detector",     supervisor)
    builder.add_conditional_edges("root_cause_finder",     supervisor)
    builder.add_conditional_edges("incident_mitigator",    supervisor)
    builder.add_conditional_edges("incident_communicator", supervisor)
    builder.add_conditional_edges("incident_summarizer",   supervisor)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
