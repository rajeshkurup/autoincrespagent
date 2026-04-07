"""Shared LangGraph state shared across all agent nodes."""

from typing import Annotated, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Workflow control ──────────────────────────────────────────────
    # Current pipeline phase; supervisor uses this to route to next node.
    # Values: "detect" | "root_cause" | "mitigate" | "feedback" |
    #         "communicate" | "summarize" | "done"
    phase: str

    # UUID for the MySQL sessions row that tracks this incident run.
    session_id: str

    # Counts feedback loop iterations (Mitigator ↔ Root Cause Finder).
    feedback_iteration: int

    # ── Incident data ─────────────────────────────────────────────────
    incident_id: Optional[str]
    severity: Optional[str]

    # Raw anomaly dicts returned by list_anomalies / root_cause_analysis.
    anomaly_nodes: List[dict]

    # Hypotheses produced by the Root Cause Finder.
    root_causes: List[dict]

    # Workflow dicts returned by the Mitigation MCP server.
    mitigation_workflows: List[dict]

    # Top cosine similarity score from the last workflow search.
    mitigation_confidence: float

    # Records of messages sent by the Communicator agent.
    communications_sent: List[dict]

    # Final human-readable story written by the Summarizer agent.
    incident_summary: Optional[str]

    # ── LangGraph message history ─────────────────────────────────────
    # Uses add_messages reducer so messages accumulate across nodes.
    messages: Annotated[list, add_messages]

    # ── Feedback channel ──────────────────────────────────────────────
    # Set by Mitigator when confidence is low; read by Root Cause Finder.
    feedback_request: Optional[str]

    # ── Communication routing ─────────────────────────────────────────
    # Set by each agent before routing to "communicate".
    # communicator prints the event then advances to next_phase.
    communication_event: Optional[str]   # "incident_detected" | "root_cause_found" | "mitigation_complete"
    next_phase: Optional[str]            # phase the communicator should advance to after printing
