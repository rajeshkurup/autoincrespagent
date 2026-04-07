"""Incident Communicator agent — Phase 4 (mock).

Invoked after each significant pipeline event:
  - incident_detected   (after Incident Detector)
  - root_cause_found    (after Root Cause Finder)
  - mitigation_complete (after Incident Mitigator)

Prints a formatted notification to stdout (no real channels — mock only).
Also upserts a rolling partial summary into Qdrant 'incident_summaries'
so the record is updated progressively throughout the incident lifecycle.

After printing, advances to state["next_phase"].
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

_LINE  = "═" * 72
_THIN  = "─" * 72

# Map event → human-readable header
_HEADERS = {
    "incident_detected":   "🚨  INCIDENT DECLARED",
    "root_cause_found":    "🔍  ROOT CAUSE IDENTIFIED",
    "mitigation_complete": "🔧  MITIGATION EXECUTED",
}

_NEXT_LABEL = {
    "root_cause": "Root Cause Analysis",
    "mitigate":   "Mitigation",
    "summarize":  "Incident Summarization",
    "done":       "Complete",
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _point_id(incident_id: str) -> str:
    """Deterministic UUID for the Qdrant summary point (stable across upserts)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"incident-summary:{incident_id}"))


def _build_partial_summary(event: str, state: AgentState) -> str:
    """Build a running plain-text summary for Qdrant (grows with each event)."""
    incident_id = state.get("incident_id", "UNKNOWN")
    severity    = state.get("severity", "UNKNOWN")
    anomalies   = state.get("anomaly_nodes", [])
    root_causes = state.get("root_causes", [])
    workflows   = state.get("mitigation_workflows", [])
    confidence  = state.get("mitigation_confidence", 0.0)

    lines = [f"Incident {incident_id} [{severity}]"]

    if event in ("incident_detected", "root_cause_found", "mitigation_complete"):
        lines.append(f"  Anomalies : {len(anomalies)} active")

    if event in ("root_cause_found", "mitigation_complete") and root_causes:
        top = root_causes[0]
        lines.append(
            f"  Top cause : {top.get('hypothesis', '?')} "
            f"({top.get('confidence', 0):.0%} confidence)"
        )

    if event == "mitigation_complete" and workflows:
        wf = workflows[0].get("payload", workflows[0])
        lines.append(f"  Workflow  : {wf.get('title', '?')} (score {confidence:.2f})")

    lines.append(f"  Updated   : {_ts()}")
    return "\n".join(lines)


async def _upsert_summary(
    qdrant_client: AsyncQdrantClient,
    embeddings: OllamaEmbeddings,
    state: AgentState,
    event: str,
) -> None:
    """Upsert the rolling summary into Qdrant incident_summaries collection."""
    incident_id = state.get("incident_id", "UNKNOWN")
    try:
        text   = _build_partial_summary(event, state)
        vector = await embeddings.aembed_query(text)
        point  = PointStruct(
            id=_point_id(incident_id),
            vector=vector,
            payload={
                "incident_id": incident_id,
                "severity":    state.get("severity", "UNKNOWN"),
                "event":       event,
                "summary":     text,
                "updated_at":  _ts(),
            },
        )
        await qdrant_client.upsert(
            collection_name="incident_summaries",
            points=[point],
        )
        logger.info(json.dumps({
            "agent": "incident_communicator", "event": "summary_upserted",
            "incident_id": incident_id, "trigger_event": event,
        }))
    except Exception as exc:
        logger.warning(f"incident_communicator: Qdrant upsert failed ({exc}) — continuing")


def make_incident_communicator(
    qdrant_client: Optional[AsyncQdrantClient] = None,
    embeddings: Optional[OllamaEmbeddings] = None,
) -> Callable:
    """Return the incident_communicator node function.

    Args:
        qdrant_client: Optional AsyncQdrantClient — enables rolling summary upserts.
        embeddings:    Optional OllamaEmbeddings  — required when qdrant_client is set.
    """

    async def incident_communicator(state: AgentState) -> dict:
        session_id  = state.get("session_id", "unknown")
        incident_id = state.get("incident_id", "UNKNOWN")
        severity    = state.get("severity", "UNKNOWN")
        event       = state.get("communication_event", "unknown")
        next_phase  = state.get("next_phase", "done")
        anomalies   = state.get("anomaly_nodes", [])
        root_causes = state.get("root_causes", [])
        workflows   = state.get("mitigation_workflows", [])
        confidence  = state.get("mitigation_confidence", 0.0)

        logger.info(json.dumps({
            "agent": "incident_communicator", "event": event,
            "incident_id": incident_id, "next_phase": next_phase,
            "session_id": session_id,
        }))

        header = _HEADERS.get(event, f"📣  {event.upper()}")

        # ── Print formatted notification ──────────────────────────────
        print(f"\n{_LINE}")
        print(f"  {header}")
        print(_LINE)

        if event == "incident_detected":
            print(f"  Incident ID  :  {incident_id}")
            print(f"  Severity     :  {severity}")
            print(f"  Anomalies    :  {len(anomalies)} active")
            if anomalies:
                for a in anomalies[:3]:
                    atype = a.get("type", a.get("id", "?"))
                    asev  = a.get("severity", "")
                    print(f"                  • {atype}" + (f" [{asev}]" if asev else ""))
                if len(anomalies) > 3:
                    print(f"                  … and {len(anomalies) - 3} more")
            # Pull reasoning out of the last AI message if present
            messages = state.get("messages", [])
            for msg in reversed(messages):
                content = getattr(msg, "content", "")
                if "Reason:" in content:
                    reason = content.split("Reason:")[-1].strip()
                    print(f"  Reason       :  {reason}")
                    break
            print(f"  Time         :  {_ts()}")
            print(f"  → Next       :  {_NEXT_LABEL.get(next_phase, next_phase)}")

        elif event == "root_cause_found":
            print(f"  Incident ID  :  {incident_id}  [{severity}]")
            if root_causes:
                top = root_causes[0]
                print(f"  Top Cause    :  {top.get('hypothesis', '?')}")
                print(f"  Node         :  {top.get('node_id', '?')} ({top.get('node_type', '?')})")
                print(f"  Confidence   :  {top.get('confidence', 0):.0%}")
                evidence = top.get("evidence", [])
                if evidence:
                    print(f"  Evidence     :  {', '.join(str(e) for e in evidence[:4])}")
                if len(root_causes) > 1:
                    print(f"  Alt causes   :  {len(root_causes) - 1} additional hypothesis/hypotheses")
            else:
                print("  Top Cause    :  (no hypotheses produced)")
            print(f"  Workflows    :  {len(workflows)} matching workflow(s) in vector DB")
            print(f"  → Next       :  {_NEXT_LABEL.get(next_phase, next_phase)}")

        elif event == "mitigation_complete":
            print(f"  Incident ID  :  {incident_id}  [{severity}]")
            if workflows:
                wf      = workflows[0]
                payload = wf.get("payload", wf)
                print(f"  Workflow     :  {payload.get('workflow_id', '?')} — {payload.get('title', '?')}")
                steps   = payload.get("steps", [])
                if steps:
                    print(f"  Steps run    :  {len(steps)}")
                    for j, step in enumerate(steps, 1):
                        print(f"                  {j}. {step}")
            else:
                print("  Workflow     :  (none applied)")
            print(f"  Confidence   :  {confidence:.2f}")
            print(f"  → Next       :  {_NEXT_LABEL.get(next_phase, next_phase)}")

        else:
            # Generic fallback for unknown events
            print(f"  Incident ID  :  {incident_id}  [{severity}]")
            print(f"  Event        :  {event}")

        print(_THIN)

        # ── Update rolling summary in Qdrant (best-effort) ────────────
        if qdrant_client and embeddings:
            await _upsert_summary(qdrant_client, embeddings, state, event)

        msg_text = f"[{event}] Communication sent for {incident_id} ({severity}). Advancing to {next_phase}."
        return {
            "phase": next_phase,
            "communication_event": None,
            "next_phase": None,
            "messages": [AIMessage(content=msg_text)],
        }

    return incident_communicator
