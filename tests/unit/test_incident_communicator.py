"""Unit tests for the incident_communicator agent node."""

import pytest
from langchain_core.messages import AIMessage
from unittest.mock import AsyncMock, MagicMock

from autoincrespagent.agents.incident_communicator import make_incident_communicator


def _state(**overrides) -> dict:
    base = {
        "phase": "communicate",
        "session_id": "test-session",
        "incident_id": "INC-TESTCOMM",
        "severity": "SEV2",
        "anomaly_nodes": [
            {"id": "ANO-001", "type": "latency_spike", "severity": "high", "status": "active"}
        ],
        "root_causes": [
            {"node_id": "db-001", "node_type": "Storage",
             "hypothesis": "Connection pool exhausted", "evidence": ["ANO-001"], "confidence": 0.9}
        ],
        "mitigation_workflows": [
            {"score": 0.85, "payload": {
                "workflow_id": "WF-001", "title": "Restart connection pool",
                "target_node_type": "Storage",
                "steps": ["Check pool metrics", "Restart pooler"],
            }}
        ],
        "mitigation_confidence": 0.85,
        "communication_event": "incident_detected",
        "next_phase": "root_cause",
        "feedback_iteration": 0,
        "communications_sent": [],
        "messages": [],
        "feedback_request": None,
        "incident_summary": None,
    }
    base.update(overrides)
    return base


class TestPhaseRouting:
    async def test_advances_to_next_phase(self):
        agent = make_incident_communicator()
        result = await agent(_state(communication_event="incident_detected", next_phase="root_cause"))
        assert result["phase"] == "root_cause"

    async def test_advances_to_mitigate_after_rca(self):
        agent = make_incident_communicator()
        result = await agent(_state(communication_event="root_cause_found", next_phase="mitigate"))
        assert result["phase"] == "mitigate"

    async def test_advances_to_summarize_after_mitigation(self):
        agent = make_incident_communicator()
        result = await agent(_state(communication_event="mitigation_complete", next_phase="summarize"))
        assert result["phase"] == "summarize"

    async def test_clears_communication_fields(self):
        agent = make_incident_communicator()
        result = await agent(_state())
        assert result["communication_event"] is None
        assert result["next_phase"] is None

    async def test_message_added(self):
        agent = make_incident_communicator()
        result = await agent(_state())
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)


class TestOutput:
    async def test_prints_incident_detected(self, capsys):
        agent = make_incident_communicator()
        await agent(_state(communication_event="incident_detected"))
        out = capsys.readouterr().out
        assert "INCIDENT DECLARED" in out
        assert "INC-TESTCOMM" in out
        assert "SEV2" in out

    async def test_prints_root_cause_found(self, capsys):
        agent = make_incident_communicator()
        await agent(_state(communication_event="root_cause_found"))
        out = capsys.readouterr().out
        assert "ROOT CAUSE" in out
        assert "Connection pool exhausted" in out
        assert "90%" in out

    async def test_prints_mitigation_complete(self, capsys):
        agent = make_incident_communicator()
        await agent(_state(communication_event="mitigation_complete"))
        out = capsys.readouterr().out
        assert "MITIGATION EXECUTED" in out
        assert "WF-001" in out
        assert "0.85" in out

    async def test_unknown_event_prints_generic(self, capsys):
        agent = make_incident_communicator()
        await agent(_state(communication_event="unknown_event"))
        out = capsys.readouterr().out
        assert "INC-TESTCOMM" in out


class TestQdrantUpsert:
    async def test_upsert_called_when_client_provided(self):
        mock_response = MagicMock()
        mock_response.points = []
        mock_client = MagicMock()
        mock_client.upsert = AsyncMock(return_value=None)
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        agent = make_incident_communicator(qdrant_client=mock_client, embeddings=mock_embeddings)
        await agent(_state(communication_event="incident_detected"))

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "incident_summaries"

    async def test_no_upsert_without_client(self):
        # No qdrant_client — must not raise
        agent = make_incident_communicator()
        result = await agent(_state())
        assert result["phase"] == "root_cause"

    async def test_upsert_failure_does_not_raise(self):
        mock_client = MagicMock()
        mock_client.upsert = AsyncMock(side_effect=Exception("qdrant down"))
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        agent = make_incident_communicator(qdrant_client=mock_client, embeddings=mock_embeddings)
        result = await agent(_state())   # must not raise
        assert result["phase"] == "root_cause"
