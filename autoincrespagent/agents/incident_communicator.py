"""Incident Communicator agent — Phase 4.

Invoked after each significant pipeline event:
  - incident_detected   (after Incident Detector)
  - root_cause_found    (after Root Cause Finder)
  - mitigation_complete (after Incident Mitigator)

For each event the agent:
  1. Selects communication channels based on incident severity.
  2. Calls the appropriate commsmcpserv tools (send_email, send_slack,
     send_teams, send_sms, page_oncall) to dispatch notifications.
  3. Calls update_ticket_comms to record the event in the ticket log.
  4. Prints a formatted summary banner to stdout.
  5. Upserts a rolling partial summary into Qdrant 'incident_summaries'.
  6. Advances to state["next_phase"].

Channel selection by severity:
  SEV1  →  page_oncall + send_slack + send_email
  SEV2  →  send_slack  + send_teams + send_email
  SEV3  →  send_email  + send_teams
  SEV4  →  send_email
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

# Channels dispatched per severity level
_SEVERITY_CHANNELS: dict[str, list[str]] = {
    "SEV1": ["page_oncall", "send_slack", "send_email"],
    "SEV2": ["send_slack", "send_teams", "send_email"],
    "SEV3": ["send_email", "send_teams"],
    "SEV4": ["send_email"],
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _point_id(incident_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"incident-summary:{incident_id}"))


def _get_tool(tools: list, name: str):
    return next((t for t in tools if t.name == name), None)


async def _call_tool(tools: list, name: str, args: dict) -> dict | None:
    """Invoke a comms MCP tool and parse the JSON response. Returns None on failure.

    langchain-mcp-adapters may return a list of content objects, a plain string,
    or already-parsed dict depending on the adapter version — handle all cases.
    """
    tool = _get_tool(tools, name)
    if not tool:
        return None
    try:
        raw = await tool.ainvoke(args)
        # Unwrap list[TextContent] → str
        if isinstance(raw, list):
            first = raw[0]
            raw = first.text if hasattr(first, "text") else (
                first.get("text", str(first)) if isinstance(first, dict) else str(first)
            )
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        logger.warning(f"incident_communicator: tool '{name}' failed — {exc}")
        return None


def _build_notification(event: str, state: AgentState) -> dict:
    """Build subject/title/body/summary strings for all channels from state."""
    incident_id = state.get("incident_id", "UNKNOWN")
    severity    = state.get("severity", "UNKNOWN")
    anomalies   = state.get("anomaly_nodes", [])
    root_causes = state.get("root_causes", [])
    workflows   = state.get("mitigation_workflows", [])
    confidence  = state.get("mitigation_confidence", 0.0)

    if event == "incident_detected":
        subject = f"[{severity}] Incident Detected: {incident_id}"
        title   = f"Incident Detected — {incident_id} [{severity}]"
        body    = (
            f"A new incident has been declared.\n\n"
            f"  Incident ID : {incident_id}\n"
            f"  Severity    : {severity}\n"
            f"  Anomalies   : {len(anomalies)} active node(s)\n"
        )
        if anomalies:
            body += "\nAffected nodes:\n"
            for a in anomalies[:5]:
                body += f"  • {a.get('type', a.get('id', '?'))} [{a.get('severity', '')}]\n"
            if len(anomalies) > 5:
                body += f"  … and {len(anomalies) - 5} more\n"
        body += f"\nDetected at: {_ts()}"
        summary_note = f"Incident {incident_id} detected with {len(anomalies)} active anomalies."

    elif event == "root_cause_found":
        top = root_causes[0] if root_causes else {}
        subject = f"[{severity}] Root Cause Identified: {incident_id}"
        title   = f"Root Cause Found — {incident_id} [{severity}]"
        body    = (
            f"Root cause analysis is complete.\n\n"
            f"  Incident ID : {incident_id}\n"
            f"  Severity    : {severity}\n"
        )
        if top:
            body += (
                f"  Top Cause   : {top.get('hypothesis', '?')}\n"
                f"  Node        : {top.get('node_id', '?')} ({top.get('node_type', '?')})\n"
                f"  Confidence  : {top.get('confidence', 0):.0%}\n"
            )
        body += f"  Workflows   : {len(workflows)} matching workflow(s) identified\n"
        body += f"\nAnalysis completed at: {_ts()}"
        summary_note = (
            f"Root cause identified for {incident_id}: "
            f"{top.get('hypothesis', 'unknown')} ({top.get('confidence', 0):.0%} confidence)."
        )

    else:  # mitigation_complete
        wf = workflows[0].get("payload", workflows[0]) if workflows else {}
        subject = f"[{severity}] Mitigation Complete: {incident_id}"
        title   = f"Mitigation Executed — {incident_id} [{severity}]"
        body    = (
            f"Mitigation steps have been executed.\n\n"
            f"  Incident ID : {incident_id}\n"
            f"  Severity    : {severity}\n"
            f"  Confidence  : {confidence:.2f}\n"
        )
        if wf:
            body += f"  Workflow    : {wf.get('workflow_id', '?')} — {wf.get('title', '?')}\n"
            steps = wf.get("steps", [])
            if steps:
                body += f"  Steps run   : {len(steps)}\n"
        body += f"\nCompleted at: {_ts()}"
        summary_note = (
            f"Mitigation complete for {incident_id}. "
            f"Workflow: {wf.get('title', 'N/A')}. Confidence: {confidence:.2f}."
        )

    return {
        "subject": subject,
        "title": title,
        "body": body,
        "summary_note": summary_note,
    }


def _build_partial_summary(event: str, state: AgentState) -> str:
    incident_id = state.get("incident_id", "UNKNOWN")
    severity    = state.get("severity", "UNKNOWN")
    anomalies   = state.get("anomaly_nodes", [])
    root_causes = state.get("root_causes", [])
    workflows   = state.get("mitigation_workflows", [])
    confidence  = state.get("mitigation_confidence", 0.0)

    lines = [f"Incident {incident_id} [{severity}]"]
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
        await qdrant_client.upsert(collection_name="incident_summaries", points=[point])
        logger.info(json.dumps({
            "agent": "incident_communicator", "event": "summary_upserted",
            "incident_id": incident_id, "trigger_event": event,
        }))
    except Exception as exc:
        logger.warning(f"incident_communicator: Qdrant upsert failed ({exc}) — continuing")


def make_incident_communicator(
    qdrant_client: Optional[AsyncQdrantClient] = None,
    embeddings: Optional[OllamaEmbeddings] = None,
    comms_tools: Optional[list] = None,
) -> Callable:
    """Return the incident_communicator node function.

    Args:
        qdrant_client: Optional AsyncQdrantClient — enables rolling summary upserts.
        embeddings:    Optional OllamaEmbeddings  — required when qdrant_client is set.
        comms_tools:   LangChain tools from commsmcpserv — enables real channel dispatch.
    """
    _tools = comms_tools or []

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
            "incident_id": incident_id, "severity": severity,
            "next_phase": next_phase, "comms_tools_available": len(_tools),
            "session_id": session_id,
        }))

        header  = _HEADERS.get(event, f"📣  {event.upper()}")
        notif   = _build_notification(event, state)
        channels_for_severity = _SEVERITY_CHANNELS.get(severity, ["send_email"])

        # ── Dispatch to communication channels via MCP ─────────────────
        dispatched: list[str] = []

        if _tools:
            for channel in channels_for_severity:
                result = None

                if channel == "send_email":
                    result = await _call_tool(_tools, "send_email", {
                        "incident_id": incident_id,
                        "subject":     notif["subject"],
                        "body":        notif["body"],
                        "recipients":  ["oncall@example.com"],
                        "severity":    severity,
                    })

                elif channel == "send_slack":
                    result = await _call_tool(_tools, "send_slack", {
                        "incident_id": incident_id,
                        "title":       notif["title"],
                        "body":        notif["body"],
                        "severity":    severity,
                        "mention":     "@here" if severity == "SEV1" else "",
                    })

                elif channel == "send_teams":
                    result = await _call_tool(_tools, "send_teams", {
                        "incident_id": incident_id,
                        "title":       notif["title"],
                        "body":        notif["body"],
                        "severity":    severity,
                    })

                elif channel == "page_oncall":
                    result = await _call_tool(_tools, "page_oncall", {
                        "incident_id": incident_id,
                        "summary":     notif["subject"],
                        "severity":    severity,
                        "details":     notif["body"],
                    })

                if result:
                    dispatched.append(channel)
                    logger.info(json.dumps({
                        "agent": "incident_communicator", "channel": channel,
                        "incident_id": incident_id,
                        "message_id": result.get("message_id", result.get("page_id", "")),
                    }))

            # Record the event + channels in the ticket log
            await _call_tool(_tools, "update_ticket_comms", {
                "incident_id":       incident_id,
                "event":             event,
                "note":              notif["summary_note"],
                "channels_notified": dispatched,
            })
        else:
            # No MCP tools — log and continue (print-only mode)
            logger.warning("incident_communicator: no comms tools available — print-only mode")

        # ── Print formatted notification banner ────────────────────────
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
            messages = state.get("messages", [])
            for msg in reversed(messages):
                content = getattr(msg, "content", "")
                if "Reason:" in content:
                    print(f"  Reason       :  {content.split('Reason:')[-1].strip()}")
                    break
            print(f"  Time         :  {_ts()}")

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
            print(f"  Workflows    :  {len(workflows)} matching workflow(s)")

        elif event == "mitigation_complete":
            print(f"  Incident ID  :  {incident_id}  [{severity}]")
            if workflows:
                wf      = workflows[0]
                payload = wf.get("payload", wf)
                print(f"  Workflow     :  {payload.get('workflow_id', '?')} — {payload.get('title', '?')}")
                steps = payload.get("steps", [])
                if steps:
                    print(f"  Steps run    :  {len(steps)}")
            else:
                print("  Workflow     :  (none applied)")
            print(f"  Confidence   :  {confidence:.2f}")

        else:
            print(f"  Incident ID  :  {incident_id}  [{severity}]")
            print(f"  Event        :  {event}")

        if dispatched:
            print(f"  Channels     :  {', '.join(dispatched)}")
        elif _tools:
            print("  Channels     :  (all dispatch calls failed)")
        else:
            print("  Channels     :  (MCP unavailable — print only)")

        print(f"  → Next       :  {_NEXT_LABEL.get(next_phase, next_phase)}")
        print(_THIN)

        # ── Upsert rolling summary to Qdrant (best-effort) ────────────
        if qdrant_client and embeddings:
            await _upsert_summary(qdrant_client, embeddings, state, event)

        # ── Update communications_sent in state ────────────────────────
        comm_record = {
            "event":      event,
            "incident_id": incident_id,
            "channels":   dispatched,
            "sent_at":    _ts(),
        }
        existing_comms = list(state.get("communications_sent") or [])
        existing_comms.append(comm_record)

        msg_text = (
            f"[{event}] Notifications dispatched for {incident_id} ({severity}) "
            f"via {', '.join(dispatched) if dispatched else 'no channels'}. "
            f"Advancing to {next_phase}."
        )
        return {
            "phase":               next_phase,
            "communication_event": None,
            "next_phase":          None,
            "communications_sent": existing_comms,
            "messages":            [AIMessage(content=msg_text)],
        }

    return incident_communicator
