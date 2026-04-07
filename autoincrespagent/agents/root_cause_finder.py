"""Root Cause Finder agent.

Synthesises evidence from three sources to produce root cause hypotheses
and pre-fetches matching mitigation workflows from the vector DB:

  1. Graph traversal  — root_cause_analysis + blast_radius + change/RCA tickets
                        via the Graph DB MCP server (graphmcpserv).
  2. Semantic search  — past RCA documents and recent change context from Qdrant.
  3. LLM synthesis    — Llama 3.1 8B combines all evidence into structured
                        hypotheses and selects mitigation workflows.

Input state fields:  anomaly_nodes, incident_id, severity, feedback_request
Output state fields: root_causes, mitigation_workflows, phase (→ "mitigate")
"""

import json
import logging
from typing import Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings
from qdrant_client import AsyncQdrantClient

from autoincrespagent.agents.state import AgentState
from autoincrespagent.llm.factory import get_llm
from autoincrespagent.vector.qdrant_search import search_collection

logger = logging.getLogger(__name__)

# ── Tools this agent needs from the Graph DB MCP ─────────────────────
_REQUIRED_TOOLS = {
    "root_cause_analysis",
    "blast_radius",
    "get_relationships",
    "get_change_tickets",
    "get_rca_tickets",
}

# ── Qdrant collections searched ───────────────────────────────────────
_COLL_RCA       = "rca_documents"
_COLL_CHANGES   = "change_context"
_COLL_MITIGATE  = "mitigation_workflows"
_COLL_SUMMARIES = "incident_summaries"
_COLL_FEEDBACK  = "feedback_history"

# ── LLM system prompt ─────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a root cause analysis engine for infrastructure incidents.

You will receive:
- Active anomalies that triggered the incident
- Graph traversal results showing topology and related anomalies
- Recent change tickets that may have introduced regressions
- Historical RCA documents from past similar incidents
- Recent change context from the knowledge base
- Past incident summaries from previous runs (if any)
- Feedback history: past low-confidence attempts and what was ultimately resolved

Your task: identify the most likely root causes and rank them by confidence.

Respond with ONLY valid JSON — no markdown, no explanation:
{
  "hypotheses": [
    {
      "node_id": "id of the suspected root cause node",
      "node_type": "Application | Storage | Network | other",
      "hypothesis": "one-sentence description of the root cause",
      "evidence": ["piece of evidence 1", "piece of evidence 2"],
      "confidence": 0.0 to 1.0
    }
  ],
  "summary": "overall one-paragraph summary of findings"
}

Rules:
- List hypotheses in descending confidence order (highest first).
- Prefer recent changes as evidence when timestamps align with anomaly start times.
- If a feedback_request is provided, use it to focus or widen your analysis.
- If evidence is sparse, lower confidence accordingly — do not fabricate.
- Maximum 5 hypotheses.
"""


# ── Internal helpers ──────────────────────────────────────────────────

def _build_search_query(
    anomaly_nodes: list[dict],
    severity: str,
    feedback_request: Optional[str],
) -> str:
    """Build a natural-language Qdrant search query from incident context."""
    types = ", ".join({a.get("type", "") for a in anomaly_nodes if a.get("type")})
    base = f"incident severity {severity} anomaly types: {types or 'unknown'}"
    if feedback_request:
        base += f". Additional context: {feedback_request}"
    return base


def _parse_json_list(raw) -> list:
    """Parse a JSON string that should be a list; return [] on failure."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _extract_to_node(rel: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract (label, id) from a relationship's 'to' field."""
    to = rel.get("to", {})
    label = to.get("label") or rel.get("toLabel")
    node_id = to.get("id") or rel.get("toId")
    return label, node_id


async def _graph_evidence(tool_map: dict, anomaly_nodes: list[dict]) -> list[dict]:
    """Run graph traversal for up to 5 anomalies and collect evidence."""
    evidence = []
    for anomaly in anomaly_nodes[:5]:
        anomaly_id = anomaly.get("id", "")
        entry: dict = {"anomaly_id": anomaly_id, "anomaly": anomaly}

        # Find the node this anomaly is detected on
        try:
            rels_raw = await tool_map["get_relationships"].ainvoke({
                "fromLabel": "Anomaly",
                "fromId": anomaly_id,
                "type": "DETECTED_ON",
            })
            rels = _parse_json_list(rels_raw)
        except Exception as exc:
            logger.warning(f"root_cause_finder: get_relationships failed for {anomaly_id}: {exc}")
            evidence.append(entry)
            continue

        for rel in rels[:2]:
            to_label, to_id = _extract_to_node(rel)
            if not (to_label and to_id):
                continue

            entry["detected_on"] = {"label": to_label, "id": to_id}

            # Root cause traversal from the affected node
            try:
                rca_raw = await tool_map["root_cause_analysis"].ainvoke({
                    "startLabel": to_label,
                    "startId": to_id,
                    "maxDepth": 5,
                    "anomalyStatus": "active",
                    "limit": 10,
                })
                entry["rca_graph"] = json.loads(rca_raw) if isinstance(rca_raw, str) else rca_raw
            except Exception as exc:
                logger.warning(f"root_cause_finder: root_cause_analysis failed for {to_id}: {exc}")

            # Blast radius — how many services depend on this node
            try:
                blast_raw = await tool_map["blast_radius"].ainvoke({
                    "label": to_label,
                    "id": to_id,
                })
                entry["blast_radius"] = json.loads(blast_raw) if isinstance(blast_raw, str) else blast_raw
            except Exception as exc:
                logger.warning(f"root_cause_finder: blast_radius failed for {to_id}: {exc}")

        evidence.append(entry)
    return evidence


async def _synthesize(
    llm,
    anomaly_nodes: list[dict],
    graph_evidence: list[dict],
    change_tickets: list[dict],
    rca_tickets: list[dict],
    past_rcas: list[dict],
    change_context: list[dict],
    feedback_request: Optional[str],
    severity: str,
    past_summaries: Optional[list[dict]] = None,
    feedback_history: Optional[list[dict]] = None,
) -> list[dict]:
    """Ask the LLM to produce structured root cause hypotheses."""
    context_parts = [
        f"Severity: {severity}",
        f"\nActive anomalies:\n{json.dumps(anomaly_nodes, indent=2)}",
        f"\nGraph traversal evidence:\n{json.dumps(graph_evidence, indent=2)}",
    ]
    if change_tickets:
        context_parts.append(f"\nRecent change tickets:\n{json.dumps(change_tickets, indent=2)}")
    if rca_tickets:
        context_parts.append(f"\nHistorical RCA tickets:\n{json.dumps(rca_tickets, indent=2)}")
    if past_rcas:
        context_parts.append(
            "\nRelevant past RCA documents (from knowledge base):\n"
            + json.dumps([h["payload"] for h in past_rcas], indent=2)
        )
    if change_context:
        context_parts.append(
            "\nRecent change context (from knowledge base):\n"
            + json.dumps([h["payload"] for h in change_context], indent=2)
        )
    if past_summaries:
        context_parts.append(
            "\nPast incident summaries (from previous runs — use to spot recurring patterns):\n"
            + json.dumps([h["payload"] for h in past_summaries], indent=2)
        )
    if feedback_history:
        context_parts.append(
            "\nFeedback history (past low-confidence attempts and resolved outcomes — "
            "avoid hypotheses marked outcome=low_confidence, prefer outcome=resolved):\n"
            + json.dumps([h["payload"] for h in feedback_history], indent=2)
        )
    if feedback_request:
        context_parts.append(f"\nFeedback from mitigator (refine analysis):\n{feedback_request}")

    try:
        response = await llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(context_parts)),
        ])
        parsed = json.loads(response.content)
        return parsed.get("hypotheses", [])
    except (json.JSONDecodeError, Exception) as exc:
        logger.error(f"root_cause_finder: LLM synthesis failed: {exc}")
        return []


# ── Public factory ────────────────────────────────────────────────────

def make_root_cause_finder(
    graph_tools: list,
    llm=None,
    qdrant_client: Optional[AsyncQdrantClient] = None,
    embeddings: Optional[OllamaEmbeddings] = None,
) -> Callable:
    """Factory that binds tools, LLM, and Qdrant client into the agent node.

    Args:
        graph_tools:   LangChain BaseTool list from MultiServerMCPClient.
        llm:           Optional LLM override (for unit tests).
        qdrant_client: Optional AsyncQdrantClient (None → skip vector search).
        embeddings:    Optional OllamaEmbeddings (None → skip vector search).

    Returns:
        Async function suitable for use as a LangGraph node.
    """
    tool_map = {t.name: t for t in graph_tools if t.name in _REQUIRED_TOOLS}
    _llm = llm if llm is not None else get_llm("root_cause_finder")

    missing = _REQUIRED_TOOLS - tool_map.keys()
    if missing:
        raise ValueError(f"make_root_cause_finder: missing tools: {missing}")

    async def root_cause_finder(state: AgentState) -> dict:
        session_id       = state.get("session_id", "unknown")
        anomaly_nodes    = state.get("anomaly_nodes", [])
        incident_id      = state.get("incident_id")
        severity         = state.get("severity", "SEV3")
        feedback_request = state.get("feedback_request")
        feedback_iter    = state.get("feedback_iteration", 0)

        logger.info(json.dumps({
            "agent": "root_cause_finder", "event": "start",
            "session_id": session_id, "anomaly_count": len(anomaly_nodes),
            "feedback_iteration": feedback_iter,
        }))

        # ── Step 1: Graph traversal ───────────────────────────────────
        graph_evidence = await _graph_evidence(tool_map, anomaly_nodes)

        # ── Step 2: Recent change + historical RCA tickets from graph ─
        change_tickets: list[dict] = []
        rca_tickets: list[dict] = []
        try:
            change_tickets = _parse_json_list(
                await tool_map["get_change_tickets"].ainvoke({"limit": 20})
            )
        except Exception as exc:
            logger.warning(f"root_cause_finder: get_change_tickets failed: {exc}")

        try:
            rca_tickets = _parse_json_list(
                await tool_map["get_rca_tickets"].ainvoke({"limit": 10})
            )
        except Exception as exc:
            logger.warning(f"root_cause_finder: get_rca_tickets failed: {exc}")

        # ── Step 3: Qdrant semantic search ────────────────────────────
        past_rcas: list[dict] = []
        change_context_hits: list[dict] = []
        past_summaries: list[dict] = []
        feedback_history: list[dict] = []
        mitigation_workflows: list[dict] = []

        if qdrant_client and embeddings:
            query = _build_search_query(anomaly_nodes, severity, feedback_request)
            past_rcas = await search_collection(
                qdrant_client, embeddings, _COLL_RCA, query, limit=5
            )
            change_context_hits = await search_collection(
                qdrant_client, embeddings, _COLL_CHANGES, query, limit=5
            )
            past_summaries = await search_collection(
                qdrant_client, embeddings, _COLL_SUMMARIES, query, limit=3
            )
            feedback_history = await search_collection(
                qdrant_client, embeddings, _COLL_FEEDBACK, query, limit=5
            )
            logger.info(json.dumps({
                "agent": "root_cause_finder", "event": "qdrant_search_done",
                "past_rcas": len(past_rcas), "change_context": len(change_context_hits),
                "past_summaries": len(past_summaries), "feedback_history": len(feedback_history),
                "session_id": session_id,
            }))

        # ── Step 4: LLM synthesises root causes ───────────────────────
        root_causes = await _synthesize(
            _llm, anomaly_nodes, graph_evidence,
            change_tickets, rca_tickets,
            past_rcas, change_context_hits,
            feedback_request, severity,
            past_summaries=past_summaries,
            feedback_history=feedback_history,
        )

        logger.info(json.dumps({
            "agent": "root_cause_finder", "event": "hypotheses_produced",
            "count": len(root_causes), "session_id": session_id,
        }))

        # ── Step 5: Fetch mitigation workflows for top hypothesis ─────
        if qdrant_client and embeddings and root_causes:
            top = root_causes[0]
            workflow_query = (
                f"{top.get('hypothesis', '')} "
                f"node_type={top.get('node_type', '')} severity={severity}"
            )
            mitigation_workflows = await search_collection(
                qdrant_client, embeddings, _COLL_MITIGATE,
                workflow_query, limit=3, score_threshold=0.3,
            )
            logger.info(json.dumps({
                "agent": "root_cause_finder", "event": "mitigation_workflows_fetched",
                "count": len(mitigation_workflows), "session_id": session_id,
            }))

        summary_msg = (
            f"Root cause analysis complete for {incident_id}. "
            f"{len(root_causes)} hypothesis/hypotheses found. "
            f"{len(mitigation_workflows)} mitigation workflow(s) retrieved."
        )

        return {
            "phase": "communicate",
            "communication_event": "root_cause_found",
            "next_phase": "mitigate",
            "root_causes": root_causes,
            "mitigation_workflows": mitigation_workflows,
            "feedback_request": None,   # clear any prior feedback
            "messages": [AIMessage(content=summary_msg)],
        }

    return root_cause_finder
