"""Incident Mitigator agent — Phase 3.

Reads root_causes and mitigation_workflows from state (produced by the
Root Cause Finder) and executes mitigation steps via the Mitigation MCP
server (mitigationmcpserv).

Each step of the top-ranked workflow is submitted through the
`execute_mitigation_step` MCP tool, which logs the action and tracks
run state.  Execution status is verified via `check_mitigation_status`.

Feedback loop:
  - If mitigation_confidence is below CONFIDENCE_THRESHOLD and the
    feedback iteration limit has not been reached, the agent requests
    another Root Cause Finder pass (phase="feedback") and stores the
    failed attempt via `store_mitigation_feedback`.
  - When the loop resolves, the winning outcome is also stored.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from langchain_core.messages import AIMessage
from langchain_ollama import OllamaEmbeddings
from qdrant_client import AsyncQdrantClient

from autoincrespagent.agents.state import AgentState
from autoincrespagent.config import settings

logger = logging.getLogger(__name__)

_LINE = "─" * 72
_COLL_FEEDBACK = "feedback_history"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_tool(tools: list, name: str):
    """Return the LangChain tool with the given name, or None."""
    return next((t for t in tools if t.name == name), None)


async def _call_tool(tools: list, name: str, args: dict) -> dict | None:
    """Invoke a mitigation MCP tool and parse the JSON response.

    Returns the parsed dict, or None on any failure.
    """
    tool = _get_tool(tools, name)
    if not tool:
        logger.warning(f"incident_mitigator: tool '{name}' not available")
        return None
    try:
        raw = await tool.ainvoke(args)
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        logger.warning(f"incident_mitigator: tool '{name}' failed — {exc}")
        return None


async def _mcp_store_feedback(
    tools: list,
    incident_id: str,
    workflow_id: str,
    run_id: str,
    outcome: str,
    query_text: str,
    notes: str = "",
) -> bool:
    """Store feedback via the MCP store_mitigation_feedback tool."""
    result = await _call_tool(tools, "store_mitigation_feedback", {
        "incident_id": incident_id,
        "workflow_id": workflow_id,
        "run_id": run_id,
        "outcome": outcome,
        "notes": notes,
        "query_text": query_text,
    })
    return bool(result and result.get("qdrant_stored"))


async def _direct_store_feedback(
    qdrant_client: AsyncQdrantClient,
    embeddings: OllamaEmbeddings,
    state: AgentState,
    outcome: str,
    confidence: float,
    feedback_msg: str = "",
) -> None:
    """Fallback: store feedback directly to Qdrant when MCP is unavailable."""
    from qdrant_client.models import Distance, PointStruct, VectorParams

    incident_id = state.get("incident_id", "UNKNOWN")
    severity    = state.get("severity", "UNKNOWN")
    root_causes = state.get("root_causes", [])
    workflows   = state.get("mitigation_workflows", [])
    iteration   = state.get("feedback_iteration", 0)

    top_rc = root_causes[0] if root_causes else {}
    top_wf_payload = {}
    if workflows:
        top_wf_payload = workflows[0].get("payload", workflows[0])

    text = (
        f"Incident {incident_id} [{severity}] feedback iteration {iteration}. "
        f"Outcome: {outcome}. Confidence: {confidence:.2f}. "
        f"Top hypothesis: {top_rc.get('hypothesis', 'none')} "
        f"(node {top_rc.get('node_id', '?')}, type {top_rc.get('node_type', '?')}). "
        f"Top workflow: {top_wf_payload.get('title', 'none')}. "
        + (f"Feedback: {feedback_msg}" if feedback_msg else "")
    )

    try:
        existing = [c.name for c in (await qdrant_client.get_collections()).collections]
        if _COLL_FEEDBACK not in existing:
            await qdrant_client.create_collection(
                collection_name=_COLL_FEEDBACK,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        vector = await embeddings.aembed_query(text)
        point  = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "incident_id":    incident_id,
                "severity":       severity,
                "outcome":        outcome,
                "iteration":      iteration,
                "confidence":     confidence,
                "hypothesis":     top_rc.get("hypothesis", ""),
                "node_id":        top_rc.get("node_id", ""),
                "node_type":      top_rc.get("node_type", ""),
                "workflow_title": top_wf_payload.get("title", ""),
                "workflow_id":    top_wf_payload.get("workflow_id", ""),
                "feedback_msg":   feedback_msg,
                "recorded_at":    _ts(),
            },
        )
        await qdrant_client.upsert(collection_name=_COLL_FEEDBACK, points=[point])
        logger.info(json.dumps({
            "agent": "incident_mitigator", "event": "feedback_saved_direct",
            "outcome": outcome, "incident_id": incident_id,
        }))
    except Exception as exc:
        logger.warning(f"incident_mitigator: direct Qdrant feedback save failed ({exc})")


def make_incident_mitigator(
    qdrant_client: Optional[AsyncQdrantClient] = None,
    embeddings: Optional[OllamaEmbeddings] = None,
    mitigation_tools: Optional[list] = None,
) -> Callable:
    """Return the incident_mitigator node function.

    Args:
        qdrant_client:    Optional AsyncQdrantClient — fallback feedback persistence.
        embeddings:       Optional OllamaEmbeddings  — required for fallback path.
        mitigation_tools: LangChain tools from mitigationmcpserv — enables real
                          step execution and MCP-based feedback storage.
    """
    _tools = mitigation_tools or []

    async def incident_mitigator(state: AgentState) -> dict:
        session_id    = state.get("session_id", "unknown")
        incident_id   = state.get("incident_id", "UNKNOWN")
        severity      = state.get("severity", "UNKNOWN")
        root_causes   = state.get("root_causes") or []
        workflows     = state.get("mitigation_workflows") or []
        feedback_iter = state.get("feedback_iteration", 0)

        logger.info(json.dumps({
            "agent": "incident_mitigator", "event": "start",
            "incident_id": incident_id, "root_causes": len(root_causes),
            "workflows": len(workflows), "feedback_iteration": feedback_iter,
            "mcp_tools_available": len(_tools),
            "session_id": session_id,
        }))

        # ── Print banner ──────────────────────────────────────────────
        print(f"\n{_LINE}")
        print(f"  INCIDENT MITIGATOR  —  {incident_id}  [{severity}]")
        print(_LINE)

        # ── Root cause summary ────────────────────────────────────────
        print("\n[ROOT CAUSES]")
        if not root_causes:
            print("  (none — root cause finder returned no hypotheses)")
        else:
            for i, rc in enumerate(root_causes, 1):
                conf  = rc.get("confidence", 0.0)
                hyp   = rc.get("hypothesis", "—")
                node  = rc.get("node_id", "?")
                ntype = rc.get("node_type", "?")
                evid  = rc.get("evidence", [])
                print(f"\n  #{i}  [{conf:.0%} confidence]  {node} ({ntype})")
                print(f"       Hypothesis : {hyp}")
                if evid:
                    print(f"       Evidence   : {', '.join(str(e) for e in evid)}")

        # ── Mitigation workflows ──────────────────────────────────────
        print("\n[MITIGATION WORKFLOWS]")
        if not workflows:
            print("  (none — no matching workflows found in vector DB)")
        else:
            for i, wf in enumerate(workflows, 1):
                payload = wf.get("payload", wf)
                score   = wf.get("score", payload.get("score", None))
                wf_id   = payload.get("workflow_id", payload.get("id", f"WF-{i}"))
                title   = payload.get("title", payload.get("name", "Untitled workflow"))
                steps   = payload.get("steps", [])
                target  = payload.get("target_node_type", "")

                score_str = f"  [similarity {score:.2f}]" if score is not None else ""
                print(f"\n  #{i}  {wf_id} — {title}{score_str}")
                if target:
                    print(f"       Target node type : {target}")
                if steps:
                    print("       Steps:")
                    for j, step in enumerate(steps, 1):
                        print(f"         {j}. {step}")
                else:
                    print("       (no steps defined)")

        # ── Determine confidence ──────────────────────────────────────
        if workflows:
            top_wf     = workflows[0]
            payload    = top_wf.get("payload", top_wf)
            confidence = top_wf.get("score", payload.get("score", 0.0)) or 0.0
        elif root_causes:
            confidence = max(rc.get("confidence", 0.0) for rc in root_causes)
        else:
            confidence = 0.0

        threshold = settings.confidence_threshold
        max_iter  = settings.max_feedback_iterations

        print(f"\n[CONFIDENCE]  {confidence:.2f}  (threshold: {threshold})")

        # ── Execute top workflow via MCP (or print if no tools) ───────
        run_id      = str(uuid.uuid4())
        query_text  = ""
        top_wf_id   = ""

        if workflows:
            top_wf      = workflows[0]
            payload     = top_wf.get("payload", top_wf)
            steps       = payload.get("steps", [])
            top_wf_id   = payload.get("workflow_id", payload.get("id", "WF-UNKNOWN"))
            wf_title    = payload.get("title", "Untitled workflow")

            # Build query text from top root cause for embedding in feedback
            if root_causes:
                rc = root_causes[0]
                query_text = (
                    f"{rc.get('hypothesis', '')} "
                    f"node {rc.get('node_id', '')} type {rc.get('node_type', '')}"
                ).strip()

            execute_tool = _get_tool(_tools, "execute_mitigation_step")

            if execute_tool:
                print(f"\n[EXECUTING — {top_wf_id}: {wf_title}]")
                if steps:
                    for j, step in enumerate(steps):
                        args = {
                            "incident_id":     incident_id,
                            "workflow_id":     top_wf_id,
                            "step_index":      j,
                            "step_description": step,
                            "run_id":          run_id,
                        }
                        result = await _call_tool(_tools, "execute_mitigation_step", args)
                        if result:
                            status = result.get("status", "unknown")
                            print(f"  step {j + 1}: {step}  [{status}]")
                        else:
                            print(f"  step {j + 1}: {step}  [tool error — logged]")

                    # Verify final run status
                    status_result = await _call_tool(
                        _tools, "check_mitigation_status", {"run_id": run_id}
                    )
                    if status_result:
                        completed = status_result.get("steps_completed", "?")
                        print(f"\n  Run {run_id}: {completed}/{len(steps)} steps completed")
                else:
                    print("  (no steps defined in workflow)")
            else:
                # MCP tool not available — print only
                print(f"\n[EXECUTION — {top_wf_id}: {wf_title}]  (MCP unavailable — display only)")
                for j, step in enumerate(steps, 1):
                    print(f"  step {j}: {step}")
        else:
            print("\n[EXECUTION]  No workflow selected — skipping.")

        print(f"\n{_LINE}\n")

        # ── Helper: store feedback via MCP or direct Qdrant ───────────
        async def _store_feedback(outcome: str, notes: str = "") -> None:
            stored = False
            if _tools:
                stored = await _mcp_store_feedback(
                    _tools, incident_id, top_wf_id, run_id,
                    outcome, query_text, notes,
                )
            if not stored and qdrant_client and embeddings:
                await _direct_store_feedback(
                    qdrant_client, embeddings, state,
                    outcome=outcome, confidence=confidence, feedback_msg=notes,
                )

        # ── Feedback loop or advance ──────────────────────────────────
        if confidence < threshold and feedback_iter < max_iter:
            feedback_msg = (
                f"Mitigation confidence {confidence:.2f} is below threshold {threshold}. "
                f"Iteration {feedback_iter + 1}/{max_iter}. "
                "Please widen root cause analysis — check upstream dependencies, "
                "recent deploys, and cross-service anomalies."
            )
            logger.info(json.dumps({
                "agent": "incident_mitigator", "event": "feedback_requested",
                "confidence": confidence, "iteration": feedback_iter + 1,
                "session_id": session_id,
            }))

            await _store_feedback("low_confidence", feedback_msg)

            return {
                "phase": "feedback",
                "feedback_iteration": feedback_iter + 1,
                "mitigation_confidence": confidence,
                "feedback_request": feedback_msg,
                "messages": [AIMessage(content=feedback_msg)],
            }

        # ── Loop resolved (or first-pass success) ─────────────────────
        logger.info(json.dumps({
            "agent": "incident_mitigator", "event": "complete",
            "confidence": confidence, "feedback_iterations": feedback_iter,
            "run_id": run_id,
            "session_id": session_id,
        }))

        if feedback_iter > 0:
            await _store_feedback("resolved")

        summary = (
            f"Mitigation plan for {incident_id} ({severity}): "
            f"{len(root_causes)} root cause(s), {len(workflows)} workflow(s) matched. "
            f"Confidence: {confidence:.2f}."
        )
        return {
            "phase": "communicate",
            "communication_event": "mitigation_complete",
            "next_phase": "summarize",
            "mitigation_confidence": confidence,
            "messages": [AIMessage(content=summary)],
        }

    return incident_mitigator
