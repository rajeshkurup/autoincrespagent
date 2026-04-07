"""Incident Mitigator agent — Phase 3 (mock).

Reads root_causes and mitigation_workflows from state (produced by the
Root Cause Finder) and prints a structured mitigation plan to stdout.
No external calls are made — this is a display-only mock that shows
what a real mitigator would execute.

If mitigation_confidence is below CONFIDENCE_THRESHOLD and the feedback
iteration limit has not been reached, it requests another Root Cause
Finder pass via phase="feedback" and saves the failed attempt to Qdrant
so future runs can avoid the same dead ends.

When the loop resolves successfully, the winning root cause + workflow
are also saved to Qdrant feedback_history.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from langchain_core.messages import AIMessage
from langchain_ollama import OllamaEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from autoincrespagent.agents.state import AgentState
from autoincrespagent.config import settings

logger = logging.getLogger(__name__)

_LINE = "─" * 72
_COLL_FEEDBACK = "feedback_history"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _save_feedback(
    qdrant_client: AsyncQdrantClient,
    embeddings: OllamaEmbeddings,
    state: AgentState,
    outcome: str,          # "low_confidence" | "resolved"
    confidence: float,
    feedback_msg: str = "",
) -> None:
    """Upsert a feedback episode into the feedback_history collection.

    outcome="low_confidence" — hypothesis failed, feedback requested.
    outcome="resolved"       — loop resolved; records what ultimately worked.
    """
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
        # Ensure collection exists (created lazily on first feedback save)
        existing = [c.name for c in (await qdrant_client.get_collections()).collections]
        if _COLL_FEEDBACK not in existing:
            await qdrant_client.create_collection(
                collection_name=_COLL_FEEDBACK,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            logger.info(f"incident_mitigator: created collection '{_COLL_FEEDBACK}'")

        vector = await embeddings.aembed_query(text)
        point  = PointStruct(
            id=str(uuid.uuid4()),   # unique per episode — never overwrite
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
            "agent": "incident_mitigator", "event": "feedback_saved",
            "outcome": outcome, "incident_id": incident_id, "iteration": iteration,
        }))
    except Exception as exc:
        logger.warning(f"incident_mitigator: feedback Qdrant save failed ({exc}) — continuing")


def make_incident_mitigator(
    qdrant_client: Optional[AsyncQdrantClient] = None,
    embeddings: Optional[OllamaEmbeddings] = None,
) -> Callable:
    """Return the incident_mitigator node function.

    Args:
        qdrant_client: Optional AsyncQdrantClient — enables feedback persistence.
        embeddings:    Optional OllamaEmbeddings  — required when qdrant_client is set.
    """

    async def incident_mitigator(state: AgentState) -> dict:
        session_id      = state.get("session_id", "unknown")
        incident_id     = state.get("incident_id", "UNKNOWN")
        severity        = state.get("severity", "UNKNOWN")
        root_causes     = state.get("root_causes") or []
        workflows       = state.get("mitigation_workflows") or []
        feedback_iter   = state.get("feedback_iteration", 0)

        logger.info(json.dumps({
            "agent": "incident_mitigator", "event": "start",
            "incident_id": incident_id, "root_causes": len(root_causes),
            "workflows": len(workflows), "feedback_iteration": feedback_iter,
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

        # ── Simulate step execution ───────────────────────────────────
        if workflows:
            top_wf  = workflows[0]
            payload = top_wf.get("payload", top_wf)
            steps   = payload.get("steps", [])
            print("\n[MOCK EXECUTION — top workflow]")
            if steps:
                for j, step in enumerate(steps, 1):
                    print(f"  ✓ step {j}: {step}  [simulated OK]")
            else:
                print("  (no steps to execute)")
        else:
            print("\n[MOCK EXECUTION]  No workflow selected — skipping.")

        print(f"\n{_LINE}\n")

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

            # Save failed attempt to Qdrant so future runs avoid this dead end
            if qdrant_client and embeddings:
                await _save_feedback(
                    qdrant_client, embeddings, state,
                    outcome="low_confidence",
                    confidence=confidence,
                    feedback_msg=feedback_msg,
                )

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
            "session_id": session_id,
        }))

        # Save winning outcome to Qdrant when feedback loop was used
        if qdrant_client and embeddings and feedback_iter > 0:
            await _save_feedback(
                qdrant_client, embeddings, state,
                outcome="resolved",
                confidence=confidence,
            )

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
