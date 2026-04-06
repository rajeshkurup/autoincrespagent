"""Incident Detector agent.

Polls graphserv for active anomalies, asks the LLM whether they
constitute an actionable incident, and — if yes — creates an
IncidentTicket via the Graph DB MCP server.

Tools used (from Graph DB MCP):
  - list_anomalies
  - create_incident_ticket
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from autoincrespagent.agents.state import AgentState
from autoincrespagent.llm.factory import get_llm

logger = logging.getLogger(__name__)

# ── Tool names this agent needs ───────────────────────────────────────
_REQUIRED_TOOLS = {"list_anomalies", "create_incident_ticket"}

# ── LLM system prompt ─────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an incident classification engine for infrastructure monitoring.

You will receive a list of active anomalies from the topology graph.
Decide whether they constitute an actionable incident that requires an on-call response.

Respond with ONLY valid JSON — no markdown, no explanation:
{
  "is_incident": true or false,
  "severity": "SEV1" or "SEV2" or "SEV3" or "SEV4",
  "reasoning": "one-sentence explanation",
  "affected_node_ids": ["list of anomaly ids that are actionable"]
}

Severity guidelines:
  SEV1 — Production outage; complete service unavailability.
  SEV2 — Significant degradation affecting multiple services or users.
  SEV3 — Partial degradation; limited user impact.
  SEV4 — Minor anomaly; worth monitoring but not urgent.

If there are no anomalies, or all are low-severity noise, set is_incident=false
and omit severity (or set it to null).
"""


def make_incident_detector(tools: list, llm=None) -> Callable:
    """Factory that binds MCP tools and the LLM into the detector node.

    Args:
        tools: List of LangChain BaseTool instances from MultiServerMCPClient.
        llm:   Optional LLM override (used in unit tests to inject FakeListChatModel).

    Returns:
        An async function suitable for use as a LangGraph node.
    """
    tool_map = {t.name: t for t in tools if t.name in _REQUIRED_TOOLS}
    _llm = llm if llm is not None else get_llm("incident_detector")

    if "list_anomalies" not in tool_map:
        raise ValueError("make_incident_detector: 'list_anomalies' tool not found in provided tools")
    if "create_incident_ticket" not in tool_map:
        raise ValueError("make_incident_detector: 'create_incident_ticket' tool not found in provided tools")

    async def incident_detector(state: AgentState) -> dict:
        session_id = state.get("session_id", "unknown")
        logger.info(json.dumps({"agent": "incident_detector", "event": "start", "session_id": session_id}))

        # ── Step 1: fetch active anomalies ────────────────────────────
        try:
            raw = await tool_map["list_anomalies"].ainvoke({"status": "active", "limit": 50})
            anomalies = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(anomalies, list):
                anomalies = []
        except Exception as exc:
            logger.error(json.dumps({
                "agent": "incident_detector", "event": "list_anomalies_failed",
                "error": str(exc), "session_id": session_id,
            }))
            return {"phase": "done", "anomaly_nodes": [], "messages": [
                AIMessage(content="incident_detector: could not reach graphserv")
            ]}

        if not anomalies:
            logger.info(json.dumps({"agent": "incident_detector", "event": "no_anomalies", "session_id": session_id}))
            return {"phase": "done", "anomaly_nodes": []}

        logger.info(json.dumps({
            "agent": "incident_detector", "event": "anomalies_found",
            "count": len(anomalies), "session_id": session_id,
        }))

        # ── Step 2: ask LLM if anomalies = incident ───────────────────
        try:
            prompt = f"Active anomalies:\n{json.dumps(anomalies, indent=2)}"
            response = await _llm.ainvoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            decision = json.loads(response.content)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error(json.dumps({
                "agent": "incident_detector", "event": "llm_failed",
                "error": str(exc), "session_id": session_id,
            }))
            return {"phase": "done", "anomaly_nodes": anomalies}

        if not decision.get("is_incident"):
            logger.info(json.dumps({
                "agent": "incident_detector", "event": "no_incident",
                "reasoning": decision.get("reasoning"), "session_id": session_id,
            }))
            return {"phase": "done", "anomaly_nodes": anomalies}

        # ── Step 3: create IncidentTicket in the graph ────────────────
        severity = decision.get("severity") or "SEV3"
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        start_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            await tool_map["create_incident_ticket"].ainvoke({
                "id": incident_id,
                "severity": severity,
                "status": "open",
                "startTime": start_time,
            })
        except Exception as exc:
            logger.error(json.dumps({
                "agent": "incident_detector", "event": "create_ticket_failed",
                "incident_id": incident_id, "error": str(exc), "session_id": session_id,
            }))
            # Detection still stands — continue even if ticket write fails.

        logger.info(json.dumps({
            "agent": "incident_detector", "event": "incident_declared",
            "incident_id": incident_id, "severity": severity,
            "anomaly_count": len(anomalies), "session_id": session_id,
        }))

        return {
            "phase": "root_cause",
            "incident_id": incident_id,
            "severity": severity,
            "anomaly_nodes": anomalies,
            "messages": [AIMessage(content=(
                f"Incident detected: {incident_id} ({severity}). "
                f"{len(anomalies)} active anomaly/anomalies. "
                f"Reason: {decision.get('reasoning', 'N/A')}"
            ))],
        }

    return incident_detector
