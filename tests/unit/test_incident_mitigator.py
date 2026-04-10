"""Unit tests for the incident_mitigator agent node."""

import json
import pytest
from langchain_core.messages import AIMessage
from unittest.mock import AsyncMock, MagicMock

from autoincrespagent.agents.incident_mitigator import make_incident_mitigator, _COLL_FEEDBACK


# ── Helpers ───────────────────────────────────────────────────────────

def _wf(score: float, steps: list | None = None) -> dict:
    """Build a minimal mitigation workflow dict as returned by Qdrant search."""
    return {
        "score": score,
        "payload": {
            "workflow_id": f"WF-{int(score * 100):03d}",
            "title": f"Workflow score={score}",
            "target_node_type": "Storage",
            "steps": steps or ["step A", "step B"],
        },
    }


def _rc(confidence: float, node_id: str = "db-001") -> dict:
    return {
        "node_id": node_id, "node_type": "Storage",
        "hypothesis": "Connection pool exhausted",
        "evidence": ["ANO-001"], "confidence": confidence,
    }


def _state(**overrides) -> dict:
    base = {
        "phase": "mitigate",
        "session_id": "test-session",
        "incident_id": "INC-TESTMIT",
        "severity": "SEV2",
        "feedback_iteration": 0,
        "root_causes": [_rc(0.9)],
        "mitigation_workflows": [_wf(0.85)],
        "mitigation_confidence": 0.0,
        "anomaly_nodes": [],
        "communications_sent": [],
        "messages": [],
        "feedback_request": None,
        "incident_summary": None,
    }
    base.update(overrides)
    return base


def _mock_mcp_tool(name: str, response: dict) -> MagicMock:
    """Build a mock LangChain tool that returns a JSON-encoded response."""
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=json.dumps(response))
    return tool


def _mock_mitigation_tools(
    execute_response: dict | None = None,
    status_response: dict | None = None,
    feedback_response: dict | None = None,
) -> list:
    """Return a list of mock mitigation MCP tools."""
    execute_resp = execute_response or {
        "run_id": "test-run-123", "status": "completed",
        "step_index": 0, "step_description": "step A", "executed_at": "2026-04-10T00:00:00Z",
    }
    status_resp = status_response or {
        "run_id": "test-run-123", "status": "in_progress", "steps_completed": 2,
        "step_log": [],
    }
    feedback_resp = feedback_response or {
        "outcome": "resolved", "qdrant_stored": True, "recorded_at": "2026-04-10T00:00:00Z",
    }
    return [
        _mock_mcp_tool("execute_mitigation_step", execute_resp),
        _mock_mcp_tool("check_mitigation_status", status_resp),
        _mock_mcp_tool("store_mitigation_feedback", feedback_resp),
    ]


# ── Factory ────────────────────────────────────────────────────────────

class TestFactory:
    def test_returns_callable(self):
        agent = make_incident_mitigator()
        assert callable(agent)

    def test_returns_callable_with_mcp_tools(self):
        agent = make_incident_mitigator(mitigation_tools=_mock_mitigation_tools())
        assert callable(agent)


# ── Phase routing ──────────────────────────────────────────────────────

class TestPhaseRouting:
    async def test_advances_to_communicate_when_confident(self):
        agent = make_incident_mitigator()
        result = await agent(_state(mitigation_workflows=[_wf(0.9)]))
        assert result["phase"] == "communicate"

    async def test_requests_feedback_when_low_confidence(self):
        agent = make_incident_mitigator()
        result = await agent(_state(mitigation_workflows=[_wf(0.3)]))
        assert result["phase"] == "feedback"

    async def test_advances_after_max_feedback_iterations(self):
        agent = make_incident_mitigator()
        result = await agent(_state(
            mitigation_workflows=[_wf(0.3)],
            feedback_iteration=3,
        ))
        assert result["phase"] == "communicate"

    async def test_advances_with_no_workflows_but_high_rc_confidence(self):
        agent = make_incident_mitigator()
        result = await agent(_state(
            mitigation_workflows=[],
            root_causes=[_rc(0.9)],
        ))
        assert result["phase"] == "communicate"

    async def test_feedback_when_no_workflows_and_low_rc_confidence(self):
        agent = make_incident_mitigator()
        result = await agent(_state(
            mitigation_workflows=[],
            root_causes=[_rc(0.4)],
            feedback_iteration=0,
        ))
        assert result["phase"] == "feedback"


# ── Confidence calculation ─────────────────────────────────────────────

class TestConfidence:
    async def test_confidence_set_from_top_workflow_score(self):
        agent = make_incident_mitigator()
        result = await agent(_state(mitigation_workflows=[_wf(0.82), _wf(0.60)]))
        assert result["mitigation_confidence"] == pytest.approx(0.82)

    async def test_confidence_falls_back_to_rc_when_no_workflows(self):
        agent = make_incident_mitigator()
        result = await agent(_state(
            mitigation_workflows=[],
            root_causes=[_rc(0.88), _rc(0.55)],
        ))
        assert result["mitigation_confidence"] == pytest.approx(0.88)

    async def test_zero_confidence_when_nothing_available(self):
        agent = make_incident_mitigator()
        result = await agent(_state(mitigation_workflows=[], root_causes=[]))
        assert result["mitigation_confidence"] == 0.0


# ── Feedback loop payload ──────────────────────────────────────────────

class TestFeedbackPayload:
    async def test_feedback_request_message_set(self):
        agent = make_incident_mitigator()
        result = await agent(_state(mitigation_workflows=[_wf(0.3)], feedback_iteration=0))
        assert result.get("feedback_request") is not None
        assert len(result["feedback_request"]) > 0

    async def test_feedback_iteration_incremented(self):
        agent = make_incident_mitigator()
        result = await agent(_state(mitigation_workflows=[_wf(0.3)], feedback_iteration=1))
        assert result["feedback_iteration"] == 2

    async def test_message_added_to_state(self):
        agent = make_incident_mitigator()
        result = await agent(_state())
        assert len(result.get("messages", [])) == 1
        assert isinstance(result["messages"][0], AIMessage)


# ── Edge cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    async def test_no_root_causes_no_workflows(self, capsys):
        agent = make_incident_mitigator()
        result = await agent(_state(root_causes=[], mitigation_workflows=[]))
        out = capsys.readouterr().out
        assert "none" in out.lower()
        assert result["phase"] == "feedback"

    async def test_handles_workflow_without_steps(self, capsys):
        wf_no_steps = {"score": 0.8, "payload": {"workflow_id": "WF-X", "title": "Empty", "steps": []}}
        agent = make_incident_mitigator()
        result = await agent(_state(mitigation_workflows=[wf_no_steps]))
        out = capsys.readouterr().out
        assert "no steps" in out.lower()

    async def test_prints_incident_id_and_severity(self, capsys):
        agent = make_incident_mitigator()
        await agent(_state(incident_id="INC-ABCD1234", severity="SEV1"))
        out = capsys.readouterr().out
        assert "INC-ABCD1234" in out
        assert "SEV1" in out


# ── MCP tool execution ─────────────────────────────────────────────────

class TestMCPExecution:
    async def test_calls_execute_for_each_step(self):
        tools = _mock_mitigation_tools()
        execute_tool = next(t for t in tools if t.name == "execute_mitigation_step")
        agent = make_incident_mitigator(mitigation_tools=tools)

        await agent(_state(mitigation_workflows=[_wf(0.9, steps=["step A", "step B", "step C"])]))

        assert execute_tool.ainvoke.call_count == 3

    async def test_execute_called_with_correct_args(self):
        tools = _mock_mitigation_tools()
        execute_tool = next(t for t in tools if t.name == "execute_mitigation_step")
        agent = make_incident_mitigator(mitigation_tools=tools)

        await agent(_state(
            incident_id="INC-MCP",
            mitigation_workflows=[_wf(0.9, steps=["step A"])],
        ))

        call_args = execute_tool.ainvoke.call_args[0][0]
        assert call_args["incident_id"] == "INC-MCP"
        assert call_args["step_index"] == 0
        assert call_args["step_description"] == "step A"

    async def test_calls_check_status_after_execution(self):
        tools = _mock_mitigation_tools()
        status_tool = next(t for t in tools if t.name == "check_mitigation_status")
        agent = make_incident_mitigator(mitigation_tools=tools)

        await agent(_state(mitigation_workflows=[_wf(0.9, steps=["step A"])]))

        assert status_tool.ainvoke.call_count == 1

    async def test_prints_step_status_from_mcp(self, capsys):
        tools = _mock_mitigation_tools(execute_response={
            "run_id": "run-abc", "status": "completed",
            "step_index": 0, "step_description": "step A", "executed_at": "2026-04-10T00:00:00Z",
        })
        agent = make_incident_mitigator(mitigation_tools=tools)
        await agent(_state(mitigation_workflows=[_wf(0.9, steps=["step A"])]))
        out = capsys.readouterr().out
        assert "completed" in out

    async def test_prints_display_only_when_no_mcp_tools(self, capsys):
        agent = make_incident_mitigator(mitigation_tools=[])
        await agent(_state(mitigation_workflows=[_wf(0.9, steps=["step A"])]))
        out = capsys.readouterr().out
        assert "mcp unavailable" in out.lower() or "display only" in out.lower()

    async def test_execute_tool_failure_does_not_raise(self):
        tools = _mock_mitigation_tools()
        execute_tool = next(t for t in tools if t.name == "execute_mitigation_step")
        execute_tool.ainvoke = AsyncMock(side_effect=Exception("mcp down"))
        agent = make_incident_mitigator(mitigation_tools=tools)

        # Should not raise even if tool fails
        result = await agent(_state(mitigation_workflows=[_wf(0.9, steps=["step A"])]))
        assert result["phase"] == "communicate"


# ── MCP feedback storage ───────────────────────────────────────────────

class TestMCPFeedback:
    async def test_stores_low_confidence_via_mcp(self):
        tools = _mock_mitigation_tools(
            feedback_response={"outcome": "low_confidence", "qdrant_stored": True}
        )
        feedback_tool = next(t for t in tools if t.name == "store_mitigation_feedback")
        agent = make_incident_mitigator(mitigation_tools=tools)

        result = await agent(_state(mitigation_workflows=[_wf(0.3)], feedback_iteration=0))

        assert result["phase"] == "feedback"
        feedback_tool.ainvoke.assert_called_once()
        call_args = feedback_tool.ainvoke.call_args[0][0]
        assert call_args["outcome"] == "low_confidence"

    async def test_stores_resolved_after_feedback_loop(self):
        tools = _mock_mitigation_tools(
            feedback_response={"outcome": "resolved", "qdrant_stored": True}
        )
        feedback_tool = next(t for t in tools if t.name == "store_mitigation_feedback")
        agent = make_incident_mitigator(mitigation_tools=tools)

        result = await agent(_state(mitigation_workflows=[_wf(0.9)], feedback_iteration=1))

        assert result["phase"] == "communicate"
        feedback_tool.ainvoke.assert_called_once()
        call_args = feedback_tool.ainvoke.call_args[0][0]
        assert call_args["outcome"] == "resolved"

    async def test_no_feedback_stored_on_first_pass_success(self):
        tools = _mock_mitigation_tools()
        feedback_tool = next(t for t in tools if t.name == "store_mitigation_feedback")
        agent = make_incident_mitigator(mitigation_tools=tools)

        await agent(_state(mitigation_workflows=[_wf(0.9)], feedback_iteration=0))

        feedback_tool.ainvoke.assert_not_called()

    async def test_falls_back_to_direct_qdrant_when_mcp_feedback_fails(self):
        tools = _mock_mitigation_tools(
            feedback_response={"outcome": "low_confidence", "qdrant_stored": False}
        )
        mock_client, mock_embeddings = _mock_qdrant()
        agent = make_incident_mitigator(
            qdrant_client=mock_client,
            embeddings=mock_embeddings,
            mitigation_tools=tools,
        )

        await agent(_state(mitigation_workflows=[_wf(0.3)], feedback_iteration=0))

        # MCP returned qdrant_stored=False → falls back to direct upsert
        mock_client.upsert.assert_called_once()


# ── Qdrant direct feedback (fallback, no MCP tools) ────────────────────

def _mock_qdrant(collection_exists=True):
    mock_collections = MagicMock()
    existing_name = MagicMock()
    existing_name.name = _COLL_FEEDBACK if collection_exists else "other"
    mock_collections.collections = [existing_name]

    mock_client = MagicMock()
    mock_client.get_collections = AsyncMock(return_value=mock_collections)
    mock_client.create_collection = AsyncMock(return_value=None)
    mock_client.upsert = AsyncMock(return_value=None)
    mock_embeddings = MagicMock()
    mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)
    return mock_client, mock_embeddings


class TestDirectQdrantFeedback:
    async def test_saves_low_confidence_directly_when_no_mcp(self):
        mock_client, mock_embeddings = _mock_qdrant()
        agent = make_incident_mitigator(qdrant_client=mock_client, embeddings=mock_embeddings)
        result = await agent(_state(mitigation_workflows=[_wf(0.3)], feedback_iteration=0))

        assert result["phase"] == "feedback"
        mock_client.upsert.assert_called_once()
        payload = mock_client.upsert.call_args[1]["points"][0].payload
        assert payload["outcome"] == "low_confidence"

    async def test_saves_resolved_directly_after_feedback_loop(self):
        mock_client, mock_embeddings = _mock_qdrant()
        agent = make_incident_mitigator(qdrant_client=mock_client, embeddings=mock_embeddings)
        result = await agent(_state(mitigation_workflows=[_wf(0.9)], feedback_iteration=1))

        assert result["phase"] == "communicate"
        mock_client.upsert.assert_called_once()
        payload = mock_client.upsert.call_args[1]["points"][0].payload
        assert payload["outcome"] == "resolved"

    async def test_no_direct_save_on_first_pass_success(self):
        mock_client, mock_embeddings = _mock_qdrant()
        agent = make_incident_mitigator(qdrant_client=mock_client, embeddings=mock_embeddings)
        await agent(_state(mitigation_workflows=[_wf(0.9)], feedback_iteration=0))

        mock_client.upsert.assert_not_called()

    async def test_qdrant_failure_does_not_raise(self):
        mock_client, mock_embeddings = _mock_qdrant()
        mock_client.upsert = AsyncMock(side_effect=Exception("qdrant down"))
        agent = make_incident_mitigator(qdrant_client=mock_client, embeddings=mock_embeddings)

        result = await agent(_state(mitigation_workflows=[_wf(0.3)]))
        assert result["phase"] == "feedback"

    async def test_no_qdrant_still_works(self):
        agent = make_incident_mitigator()
        result = await agent(_state(mitigation_workflows=[_wf(0.3)]))
        assert result["phase"] == "feedback"
