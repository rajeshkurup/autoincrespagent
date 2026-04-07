"""Incident Summarizer agent — Phase 5 (final).

Composes a complete human-readable incident summary from the full
AgentState and:
  1. Prints the summary to stdout.
  2. Upserts the summary into the Qdrant 'incident_summaries' collection
     (same deterministic point ID used by the communicator, so this is
     the final authoritative upsert overwriting the partial ones).

No LLM call is made — the summary is built deterministically from state
so it is always consistent and never hallucinates.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from langchain_core.messages import AIMessage
from langchain_ollama import OllamaEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from autoincrespagent.agents.state import AgentState

logger = logging.getLogger(__name__)

_LINE = "═" * 72
_THIN = "─" * 72


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _point_id(incident_id: str) -> str:
    """Deterministic UUID — same as communicator so upserts replace the same point."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"incident-summary:{incident_id}"))


def _compose_summary(state: AgentState) -> str:
    """Build a complete plain-text incident summary from state."""
    incident_id = state.get("incident_id", "UNKNOWN")
    severity    = state.get("severity",    "UNKNOWN")
    anomalies   = state.get("anomaly_nodes",       [])
    root_causes = state.get("root_causes",         [])
    workflows   = state.get("mitigation_workflows",[])
    confidence  = state.get("mitigation_confidence", 0.0)
    feedback_it = state.get("feedback_iteration",  0)

    parts = [f"INCIDENT SUMMARY — {incident_id} [{severity}]", ""]

    # ── Anomalies ──────────────────────────────────────────────────────
    parts.append(f"Anomalies detected: {len(anomalies)}")
    for a in anomalies:
        atype = a.get("type", a.get("id", "?"))
        asev  = a.get("severity", "")
        astat = a.get("status", "")
        parts.append(f"  • {atype}" + (f" [{asev}]" if asev else "") + (f" status={astat}" if astat else ""))

    parts.append("")

    # ── Root causes ────────────────────────────────────────────────────
    parts.append(f"Root cause hypotheses: {len(root_causes)}")
    for i, rc in enumerate(root_causes, 1):
        parts.append(
            f"  #{i} [{rc.get('confidence', 0):.0%}]  "
            f"{rc.get('node_id', '?')} ({rc.get('node_type', '?')}): "
            f"{rc.get('hypothesis', '?')}"
        )
        ev = rc.get("evidence", [])
        if ev:
            parts.append(f"       Evidence: {', '.join(str(e) for e in ev)}")

    parts.append("")

    # ── Mitigation ─────────────────────────────────────────────────────
    parts.append(f"Mitigation confidence: {confidence:.2f}")
    if feedback_it:
        parts.append(f"Feedback loop iterations: {feedback_it}")
    parts.append(f"Workflows matched: {len(workflows)}")
    for i, wf in enumerate(workflows, 1):
        payload = wf.get("payload", wf)
        score   = wf.get("score", None)
        title   = payload.get("title", payload.get("name", "?"))
        wfid    = payload.get("workflow_id", "?")
        score_s = f" (score {score:.2f})" if score is not None else ""
        parts.append(f"  #{i} {wfid} — {title}{score_s}")
        for step in payload.get("steps", []):
            parts.append(f"       • {step}")

    parts.append("")
    parts.append(f"Summary generated at: {_ts()}")

    return "\n".join(parts)


def make_incident_summarizer(
    qdrant_client: Optional[AsyncQdrantClient] = None,
    embeddings: Optional[OllamaEmbeddings] = None,
) -> Callable:
    """Return the incident_summarizer node function.

    Args:
        qdrant_client: Optional AsyncQdrantClient for persisting the summary.
        embeddings:    Optional OllamaEmbeddings — required when qdrant_client is set.
    """

    async def incident_summarizer(state: AgentState) -> dict:
        session_id  = state.get("session_id", "unknown")
        incident_id = state.get("incident_id", "UNKNOWN")
        severity    = state.get("severity", "UNKNOWN")

        logger.info(json.dumps({
            "agent": "incident_summarizer", "event": "start",
            "incident_id": incident_id, "session_id": session_id,
        }))

        summary_text = _compose_summary(state)

        # ── Print to stdout ───────────────────────────────────────────
        print(f"\n{_LINE}")
        print(f"  📋  INCIDENT SUMMARY")
        print(_LINE)
        print()
        for line in summary_text.splitlines():
            print(f"  {line}")
        print()
        print(_THIN)

        # ── Persist final summary to Qdrant ───────────────────────────
        if qdrant_client and embeddings:
            try:
                vector = await embeddings.aembed_query(summary_text)
                point  = PointStruct(
                    id=_point_id(incident_id),
                    vector=vector,
                    payload={
                        "incident_id":          incident_id,
                        "severity":             severity,
                        "session_id":           session_id,
                        "summary":              summary_text,
                        "anomaly_count":        len(state.get("anomaly_nodes", [])),
                        "root_cause_count":     len(state.get("root_causes", [])),
                        "workflow_count":       len(state.get("mitigation_workflows", [])),
                        "mitigation_confidence":state.get("mitigation_confidence", 0.0),
                        "feedback_iterations":  state.get("feedback_iteration", 0),
                        "completed_at":         _ts(),
                    },
                )
                await qdrant_client.upsert(
                    collection_name="incident_summaries",
                    points=[point],
                )
                logger.info(json.dumps({
                    "agent": "incident_summarizer", "event": "summary_saved",
                    "incident_id": incident_id, "session_id": session_id,
                }))
            except Exception as exc:
                logger.warning(f"incident_summarizer: Qdrant upsert failed ({exc}) — continuing")

        logger.info(json.dumps({
            "agent": "incident_summarizer", "event": "complete",
            "incident_id": incident_id, "session_id": session_id,
        }))

        return {
            "phase": "done",
            "incident_summary": summary_text,
            "messages": [AIMessage(content=f"Incident {incident_id} fully summarized and closed.")],
        }

    return incident_summarizer
