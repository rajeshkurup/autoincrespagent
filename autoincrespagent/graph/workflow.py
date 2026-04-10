"""StateGraph assembly.

Wires agent nodes and conditional edges together into a compiled graph.
Implemented agents: incident_detector, root_cause_finder.
Remaining nodes are stubs that will be replaced as each phase is built out.
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from autoincrespagent.agents.incident_communicator import make_incident_communicator
from autoincrespagent.agents.incident_detector import make_incident_detector
from autoincrespagent.agents.incident_mitigator import make_incident_mitigator
from autoincrespagent.agents.incident_summarizer import make_incident_summarizer
from autoincrespagent.agents.root_cause_finder import make_root_cause_finder
from autoincrespagent.agents.state import AgentState
from autoincrespagent.agents.supervisor import supervisor
from autoincrespagent.vector.qdrant_search import build_embeddings, build_qdrant_client

logger = logging.getLogger(__name__)

# All tool names served by the Graph DB MCP server
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

# Tool names served by the Mitigation MCP server
_MITIGATION_TOOL_NAMES = {
    "search_mitigation_workflows",
    "execute_mitigation_step",
    "check_mitigation_status",
    "store_mitigation_feedback",
}

# Tool names served by the Communication MCP server
_COMMS_TOOL_NAMES = {
    "send_email",
    "send_slack",
    "send_teams",
    "send_sms",
    "page_oncall",
    "update_ticket_comms",
}


# ── Graph builder ─────────────────────────────────────────────────────

def build_graph(all_tools: list, checkpointer=None):
    """Compile and return the incident response StateGraph.

    Args:
        all_tools:    List of LangChain BaseTool instances from the MCP client.
        checkpointer: LangGraph checkpointer (defaults to MemorySaver for dev).

    Returns:
        A compiled LangGraph graph ready for ainvoke / astream.
    """
    graph_tools      = [t for t in all_tools if t.name in _GRAPH_DB_TOOL_NAMES]
    mitigation_tools = [t for t in all_tools if t.name in _MITIGATION_TOOL_NAMES]
    comms_tools      = [t for t in all_tools if t.name in _COMMS_TOOL_NAMES]
    logger.info(f"build_graph: loaded {len(graph_tools)} graph DB tools, "
                f"{len(mitigation_tools)} mitigation tools, "
                f"{len(comms_tools)} comms tools")

    # Try to connect to Qdrant; agents skip vector search if unavailable
    try:
        qdrant_client = build_qdrant_client()
        embeddings = build_embeddings()
        logger.info("build_graph: Qdrant client initialised")
    except Exception as exc:
        logger.warning(f"build_graph: Qdrant unavailable ({exc}) — vector search disabled")
        qdrant_client = None
        embeddings = None

    builder = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────
    builder.add_node("incident_detector",
                     make_incident_detector(graph_tools))
    builder.add_node("root_cause_finder",
                     make_root_cause_finder(graph_tools, qdrant_client=qdrant_client, embeddings=embeddings))
    builder.add_node("incident_mitigator",
                     make_incident_mitigator(qdrant_client=qdrant_client, embeddings=embeddings,
                                             mitigation_tools=mitigation_tools))
    builder.add_node("incident_communicator",
                     make_incident_communicator(qdrant_client=qdrant_client, embeddings=embeddings,
                                               comms_tools=comms_tools))
    builder.add_node("incident_summarizer",
                     make_incident_summarizer(qdrant_client=qdrant_client, embeddings=embeddings))

    # ── Entry point ───────────────────────────────────────────────────
    builder.set_entry_point("incident_detector")

    # ── Conditional edges (all route through supervisor) ──────────────
    builder.add_conditional_edges("incident_detector",     supervisor)
    builder.add_conditional_edges("root_cause_finder",     supervisor)
    builder.add_conditional_edges("incident_mitigator",    supervisor)
    builder.add_conditional_edges("incident_communicator", supervisor)
    builder.add_conditional_edges("incident_summarizer",   supervisor)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
